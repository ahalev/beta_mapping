from beta_constraints.mapping.beta_mapping import ConstraintMapping
import torch.nn as nn
import torch
import numpy as np
from itertools import combinations, product

class TorchConstraintMapping(nn.Module, ConstraintMapping):
    def __init__(self, env, safety_layer):
        super().__init__()
        super(nn.Module, self).__init__(env, safety_layer)

    def forward(self, obs, beta_action, c):
        return self.map_beta_to_safe(obs, beta_action, c)

    def get_constraint_boundaries(self, observation, c, augment_action_space=True):
        c, g = self.safety_layer.get_constraint_boundaries(observation,
                                                           c,
                                                           augment_action_space=augment_action_space,
                                                           leq_zero=True)
        return c, g

    def get_p_faster(self, A, C):
        # TODO two things here:
        # 1. This is mapping all constraints at once. Do we want to do that? If we do it one by one, we have more control
        # 2. Currently it maps to the boundary of a constraint. You should go farther, but probably need some decay factor
        # to ensure it doesn't overrun all the other dimensions eventually.

        all_at_once=False # if all_at_once, projects to each constraint at the same time. Note that b/c constraints aren't
                        # orthogonal, it can cause the overall projection to still not satisfy some constraints.

        c_action_space, g_action_space = self.get_action_space_constraints()
        x_i_corners = [(g_action_space[:, i] / c_action_space)[g_action_space[:, i] != 0] for i in range(self.env.action_space.shape[0])]
        candidate_p = torch.as_tensor([x_i.mean() for x_i in x_i_corners], dtype=torch.float32)

        if all_at_once:
            sq_norm_of_each_row_A = torch.linalg.norm(A, dim=1)**2
            scale_factors = (C - A @ candidate_p) / sq_norm_of_each_row_A
            negative_scale_factors = torch.where(scale_factors < 0, scale_factors, torch.zeros(1))
            while (negative_scale_factors < 0).any():
                candidate_p += scale_factors@A
                norm_of_each_row_A = torch.linalg.norm(A, dim=1)
                scale_factors = (C - A @ candidate_p) / norm_of_each_row_A
                negative_scale_factors = torch.where(scale_factors < 0, scale_factors, 0.0)
        else:
            while (A@candidate_p > C).any():
                for i in range(self.n_constraints):
                    row_sq_norm = (A[i, :]**2).sum()
                    scale_factor = (C[i]-torch.dot(A[i,:], candidate_p))/row_sq_norm
                    negative_scale_factors = torch.minimum(scale_factor, 0.0)
                    candidate_p += negative_scale_factors*A[i, :]

        assert (A@candidate_p <= C).all()
        return candidate_p

    def get_p(self, A, C):
        """
        A@action<=C
        :param A:
        :param C:
        :return:
        """
        self.get_p_faster(A, C)

        n, m = A.shape
        # num_possible_corners = comb(n,m)
        # action_space_corners = 2**m
        corners = []
        unsatisfiable  = []

        for j, rows in enumerate(combinations(range(n), m)):
            corner = self._compute_corner(A, C, rows)
            if isinstance(corner, tuple):
                if rows[0] >= self.n_constraints and rows[1] >= self.n_constraints:
                    unsatisfiable.append(corner[1])
            elif corner is not None:
                corners.append(corner)
        if len(corners) == 0:
            # At least one constraint didn't work. Gotta remove it
            failures = []
            for k in unsatisfiable[0]:
                to_append = True
                for unsatisfied in unsatisfiable[1:]:
                    if k not in unsatisfied:
                        to_append = False
                        break
                if to_append:
                    failures.append(k)

            print('Constraint(s) {} unsatisfiable, removing it'.format(failures))
            self.A = np.delete(A, failures, axis=0)
            self.C = np.delete(C, failures, axis=0)
            return self.get_p(self.A, self.C)

        corners = torch.stack(corners)
        p = torch.mean(corners, axis=0)

        assert p.shape == (m,)
        return p, corners

    def _compute_corner(self, A, C, rows):
        a = A[rows, :]
        c = C[rows, ]

        """  # Old way, more stable:
        if np.linalg.cond(a) < 1 / sys.float_info.epsilon: # Make sure none of the constraint boundaries are parallel
            corner = np.linalg.solve(a, c)
            if np.logical_or(A@corner-C <= 0, np.isclose(A@corner-C, 0)).all(): # Make sure this corner satisfies all the constraints
                return corner
            else:
                return None
        """
        # New way
        try:
            corner = torch.linalg.solve(a, c)
            if (A@corner-C <= 0).all():
                return corner
            else:
                rows_arr = np.array([False if j not in rows else True for j in range(A.shape[0])])
                constraints_violated = np.where(~(A@corner-C <= 0) & ~rows_arr)
                return False, constraints_violated[0]
        except RuntimeError:
            return None

    def get_D(self, A, C):
        g, c = A, -C
        alpha = np.infty
        shape = self.env.action_space.shape[0]

        for alpha_coef in product((-1, 1), repeat=shape):  # For each corner
            coef = torch.FloatTensor(alpha_coef)
            for i, c_i in enumerate(c):  # For each constraint
                alpha_ij = -(c_i + torch.dot(g[i, :], self.p)) / torch.dot(g[i, :], coef)
                if torch.abs(alpha_ij) < alpha:
                    alpha = torch.abs(alpha_ij)
        return alpha

    def map_to_D(self, action, alpha):
        """
        Assumes action is in [0,1]^n
        :param action:
        :param alpha:
        :return:
        """
        if not isinstance(action, torch.Tensor):
            # TODO you eventually need to make this so that this action is passed from obs, not selected separately
            # E.g. you need obs->policy->beta_action
            #               |                  |
            #               ->                 ->       scaled_action
            action = torch.FloatTensor(action)

        assert (action >= 0).all()
        assert (action <= 1).all()

        action = action-0.5*torch.ones(action.shape) # Map to [-1/2, 1/2]^n
        action *= 2*alpha       # Map to alpha*[-1, 1]^n
        action += self.p           # Map to p+alpha*[-1,1]^n

        return action

    def compute_collinear_point_on_boundary(self, action, A, C):
        """
        This is the b_j on your notebook
        Constraints below are of the form A@action<=C
        :return:
        """
        beta = []

        for j in range(A.shape[0]):
            t = (C[j]-torch.dot(A[j, :], action))/torch.dot(A[j, :], self.p-action)
            beta_j = t*self.p+(1-t)*action
            beta.append(beta_j)

        direction = torch.sign(action-self.p)
        distances = [torch.sum((b_j-action)**2) for b_j in beta]

        min_distance = np.infty
        bounding_beta = None

        # TODO I think you need to write this as a torch function
        for j, b_j in enumerate(beta):
            if (torch.isclose(b_j, action).all() or (torch.sign(b_j-action) == direction).all()) and distances[j] < min_distance:
                bounding_beta = b_j
                min_distance = distances[j]

        if min_distance == np.infty or bounding_beta is None:
            raise RuntimeError('Something is wrong, min_distance is infinity and did not find bounding beta')

        return bounding_beta

    def compute_d(self, alpha, bounding_beta):
        """
        Compute the point d that is collinear with p and bounding_beta and on the boundary of D
        :return:
        """
        d_plus = self.p+alpha
        d_minus = self.p-alpha

        t_plus = (d_plus-self.p)/(-self.p+bounding_beta)
        t_minus = (d_minus-self.p)/(-self.p+bounding_beta)

        t_pm = torch.cat((t_plus, t_minus))
        potential_min_idx = torch.where(torch.isclose(torch.abs(t_pm), torch.abs(t_pm).min()))[0]

        for j, idx in enumerate(potential_min_idx):
            t = t_pm[idx]
            d = t*bounding_beta+(1-t)*self.p
            if (torch.sign(d-self.p) == torch.sign(bounding_beta-self.p)).all():
                break
            if j == len(potential_min_idx)-1:
                raise RuntimeError('Something is wrong, couldn\'t find correct point on boundary of D')

        if t > 1 and torch.isclose(t, torch.FloatTensor([1.0])):
            t = 1.0
        elif t < -1 and torch.isclose(t, torch.FloatTensor([-1.0])):
            t = -1.0

        assert -1 <= t <= 1, 't should be in range[-1,1], is {t}'

        return d

    def scale_action(self, beta_action, d, bounding_beta):
        lam = torch.divide(beta_action-self.p, d-self.p)
        lam[torch.isclose(d,self.p)] = d[torch.isclose(d,self.p)]
        constrained_region_action = lam*(bounding_beta-self.p)+self.p
        return constrained_region_action
