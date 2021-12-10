from gym import Env
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import patches as patches
from itertools import product
from beta_constraints.plotting import plot_world_outlines
from itertools import combinations
import sys
import torch
from scipy.special import comb

class ConstraintMapping:
    """
    A class to pass an action taken from a beta distribution on [0,1] to a convex region defined by
    constraints.
    # TODO you need a check to see if the point to map is p. Should just return p in that case.
    """

    def __init__(self, env, safety_layer):
        assert(isinstance(env, Env) or (hasattr(env, 'step') and hasattr(env, 'reset')))
        self.env = env
        self.safety_layer = safety_layer
        self.p = None

        if hasattr(self.env, 'get_num_constraints'):
            self.n_constraints = self.env.get_num_constraints()
        else:
            raise AttributeError('env must be a constrained gym Env with a get_num_constraints method')

        assert all(self.env.action_space.high<np.infty), 'Action space must be finite'
        assert all(self.env.action_space.low>-np.infty), 'Action space must be finite'

    def get_p(self, A, C):
        """
        A@action<=C
        :param A:
        :param C:
        :return:
        """
        n, m = A.shape
        num_possible_corners = comb(n,m)
        action_space_corners = 2**m
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

        corners = np.array(corners)
        p = np.mean(corners, axis=0)

        assert p.shape == (m,)
        return p, corners

    def _old_compute_corner(self, A, C, rows):
        a = A[rows, :]
        c = C[rows, ]

        # Old way, more stable:
        if np.linalg.cond(a) < 1 / sys.float_info.epsilon: # Make sure none of the constraint boundaries are parallel
            corner = np.linalg.solve(a, c)
            if np.logical_or(A@corner-C <= 0, np.isclose(A@corner-C, 0)).all(): # Make sure this corner satisfies all the constraints
                return corner
            else:
                return None

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
            corner = np.linalg.solve(a, c)
            if (A@corner-C <= 0).all():
                return corner
            else:
                rows_arr = np.array([False if j not in rows else True for j in range(A.shape[0])])
                constraints_violated = np.where(~(A@corner-C <= 0) & ~rows_arr)
                return False, constraints_violated[0]
        except np.linalg.LinAlgError:
            return None

    def check_constraint_satisfaction(self, point, c, g=None, obs=None):
        if g is None and obs is None:
            raise RuntimeError('Must pass g or obs')
        if g is None:
            c, g = self.get_constraint_boundaries(obs, c)
        vals = c+g@point
        satisfied = np.all(vals<=0)
        if satisfied:
            print('All constraints satisfied according to safety layer')
        else:
            print('Not all constraints satisfied according to safety layer')
        return satisfied

    def check_constraint_prediction_error(self, point, c, c_next, g=None, obs=None):
        if g is None and obs is None:
            raise RuntimeError('Must pass g or obs')
        if g is None:
            c, g = self.get_constraint_boundaries(obs, c, augment_action_space=False)
        vals = c+g@point-c_next
        return vals


    def map_to_D(self, action, alpha):
        """
        Assumes action is in [0,1]^n
        :param action:
        :param alpha:
        :return:
        """
        for v in action:
            assert 0 <= v <= 1

        action = action-0.5*np.ones(action.shape) # Map to [-1/2, 1/2]^n
        action *= 2*alpha       # Map to alpha*[-1, 1]^n
        action += self.p           # Map to p+alpha*[-1,1]^n

        return action

    def get_D(self, A, C):
        """
        Returns the alpha that defines the region D defined by p+alpha[-1,1]^n
        Here, c is -C and g is A from your notebook, e.g. constraints are of the form c+g@a<=0
        :param observation:
        :param c:
        :return:
        """
        g, c = A, -C
        alpha = np.infty
        shape = self.env.action_space.shape[0]

        # TODO this is running twice as many as you need, as e.g. (1,-1,1) and (-1,1,-1) are same under alpha->-alpha
        for alpha_coef in product((-1, 1), repeat=shape):  # For each corner
            for i, c_i in enumerate(c):  # For each constraint
                alpha_ij = -(c_i + np.dot(g[i, :], self.p)) / np.dot(g[i, :], alpha_coef)
                if np.abs(alpha_ij) < alpha:
                    alpha = np.abs(alpha_ij)
        return alpha


    def get_action_space_constraints(self):
        """
        # TODO this is inefficient, you can save these vals and not recompute
        Gets action space constraints of the form c+g@a<=0
        :return:
        """
        return self.safety_layer.get_action_space_constraints()

    def get_constraint_boundaries(self, observation, c, augment_action_space=True):
        """
        Returns c, g such that the constrained region is defined by c+g@a <= 0
        :param observation:
        :param c: Should be the current constraint values
        :return:
        """

        c, g = self.safety_layer.get_constraint_boundaries(observation,
                                                           c,
                                                           augment_action_space=augment_action_space,
                                                           leq_zero=True)
        return c.detach().numpy(), g.detach().numpy()
        # return c, g
        try:
            g = [x(self.safety_layer._as_tensor(observation["agent_position"]).view(1, -1)) for x in self.safety_layer._models]
        except IndexError:
            g = [x(self.safety_layer._as_tensor(observation).view(1, -1)) for x in self.safety_layer._models]
        g = np.array([x.data.numpy().reshape(-1) for x in g])

        if augment_action_space:
            c_action_space, g_action_space = self.get_action_space_constraints()
            c = np.concatenate((c,c_action_space))
            g = np.concatenate((g, g_action_space))

        return c, g

    def compute_collinear_point_on_boundary(self, action, A, C):
        """
        This is the b_j on your notebook
        Constraints below are of the form A@action<=C
        :return:
        """
        beta = []

        for j in range(A.shape[0]):
            t = (C[j]-np.dot(A[j, :], action))/np.dot(A[j, :], self.p-action)
            beta_j = t*self.p+(1-t)*action
            beta.append(beta_j)

        direction = np.sign(action-self.p)
        distances = [np.sum((b_j-action)**2) for b_j in beta]

        min_distance = np.infty
        bounding_beta = None
        for j, b_j in enumerate(beta):
            if (np.isclose(b_j, action).all() or (np.sign(b_j-action) == direction).all()) and distances[j] < min_distance:
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

        t_pm = np.concatenate((t_plus, t_minus))
        potential_min_idx = np.where(np.isclose(np.abs(t_pm), np.abs(t_pm).min()))[0]

        for j, idx in enumerate(potential_min_idx):
            t = t_pm[idx]
            d = t*bounding_beta+(1-t)*self.p
            if (np.sign(d-self.p) == np.sign(bounding_beta-self.p)).all():
                break
            if j == len(potential_min_idx)-1:
                raise RuntimeError('Something is wrong, couldn\'t find correct point on boundary of D')

        if t>1 and np.isclose(t, 1.0):
            t = 1.0
        elif t<-1 and np.isclose(t, -1.0):
            t = -1.0

        assert -1 <= t <= 1, 't should be in range[-1,1], is {t}'

        return d

    def scale_action(self, beta_action, d, bounding_beta):
        lam = np.divide(beta_action-self.p, d-self.p, out = d.copy(), where=~np.isclose(d,self.p))
        constrained_region_action = lam*(bounding_beta-self.p)+self.p
        return constrained_region_action

    @staticmethod
    def _is_close(a, b):
        try:
            return torch.isclose(a, b)
        except TypeError:
            return np.isclose(a, b)

    def map_beta_to_safe(self, obs, beta_action, c, plot=False, verbose=False, save_fig_name=None):
        """
        Map an action selected from the beta distribution through an injective function to the safe region
        # TODO if there is an error here, make sure obs and beta_action are in correct order (you flipped them)
        :return:
        """
        c, g = self.get_constraint_boundaries(obs, c) # c+g@a<=0
        self.A, self.C = g, -c                                            # A@action<=c
        self.p, corners = self.get_p(self.A, self.C)
        alpha = self.get_D(self.A, self.C)
        translated_action = self.map_to_D(beta_action, alpha)

        if self._is_close(self.p, translated_action).all():
            scaled_action = translated_action
            d, bounding_beta = None, None
        else:
            bounding_beta = self.compute_collinear_point_on_boundary(translated_action, self.A, self.C)
            d = self.compute_d(alpha, bounding_beta)
            scaled_action = self.scale_action(translated_action, d, bounding_beta)
        if plot:
            self.plot_mapping(beta_action=translated_action, scaled_action=scaled_action, alpha=alpha, d=d, A=self.A, C=self.C,
                         bounding_beta=bounding_beta, corners=corners, fig_name=save_fig_name)

        if verbose:
            print('Original: {}; translated: {}; scaled: {}'.format(beta_action, translated_action, scaled_action))

        return scaled_action

    def plot_mapping(self, beta_action=None,
                     scaled_action=None,
                     d=None,
                     A=None,
                     C=None,
                     alpha=None,
                     bounding_beta=None,
                     corners=None,
                     region_size_factor=1.,
                     fig_name=None):
        """
        A@action<=C
        :param beta_action:
        :param A:
        :param C:
        :return:
        """
        plt.figure()
        max_ac_bound = np.max(np.concatenate((np.abs(self.env.action_space.high), np.abs(self.env.action_space.low))))
        min_ac_bound = np.min(np.concatenate((-np.abs(self.env.action_space.high), -np.abs(self.env.action_space.low))))
        region_size = region_size_factor*max_ac_bound

        # Plot the interpolating line between p and everything
        if self.p is not None and bounding_beta is not None:
            self.func_plot(lambda t: np.array([t_i*self.p+(1-t_i)*bounding_beta for t_i in t]),
                           bounds=(-0.25, 1.25), label=None)
        elif self.p is not None and beta_action is not None:
            self.func_plot(lambda t: np.array([t_i*self.p+(1-t_i)*bounding_beta for t_i in t]),
                           bounds=(min_ac_bound/2, max_ac_bound/2), label='Interpolating line')

        # Plot D:
        if alpha is not None:
            self.plot_d(alpha)

        if A is not None and C is not None:
            self.plot_constraint_region(A, C, region_size=region_size)
        if (A is None) != (C is None):
            raise ValueError('Passed one constraint region boundary array but not the other')

        # Plot the centerpoint p
        if self.p is not None:
            plt.plot(self.p[0], self.p[1], marker='.', color='b', fillstyle='full')

        # Plot the corners from which p was determined
        if corners is not None:
            plt.scatter(corners[:, 0], corners[:, 1], facecolors='b', marker='.', edgecolors='b', zorder=3)

        # Plot the action selected from the beta distribution
        if beta_action is not None:
            plt.scatter(beta_action[0], beta_action[1], marker='.', label='Beta Action', color='red', zorder=3)

        # Plot the point on the boundary of the constraint region that is collinear with p and beta_action
        if bounding_beta is not None:
            plt.plot(bounding_beta[0], bounding_beta[1], marker='.', fillstyle='full', color='c')

        # Plot the point on the boundary of the d that is collinear with everything else
        if d is not None:
            plt.plot(d[0], d[1], marker='.', fillstyle='full', color='c')

        # Plot the action scaled to the constraint region
        if scaled_action is not None:
            plt.scatter(scaled_action[0], scaled_action[1], marker='o', label='Safe Action', color='r', zorder=3)

        plt.legend()
        if fig_name is not None:
            plt.savefig(fig_name, transparent=True, bbox_inches = 'tight', pad_inches = 0)
        plt.show(block=True)

    def func_plot(self, func, bounds=(-2,2), label=None):
        """
        Plots the parametric function func over parameter values between bounds[0] and bounds[1].
        :param func:
        :param bounds:
        :param label:
        :return:
        """
        t = np.linspace(bounds[0], bounds[1])
        vals = func(t)
        plt.plot(vals[:,0], vals[:,1], '--', label=label, color='c')

    def plot_constraint_region(self, A, C, region_size=5):
        """
        A@action<=C
        :param A:
        :param C:
        :return:
        """
        constraints = []
        num_points = 1000
        d = np.linspace(-region_size, region_size, num_points)
        x,y = np.meshgrid(d, -d)

        for j in range(len(C)):
            v1 =  A[j,0]
            v2 = A[j,1]
            v3 = C[j]

            f = lambda x, y: v1*x+v2*y-v3
            constraints.append(f(x, y) <= 0)

        where_safe = np.all(constraints, axis=0).astype(int)

        plot_world_outlines(where_safe, extent=(-region_size, region_size, -region_size, region_size), num_points=num_points)

    def plot_d(self, alpha, ax=None):
        if ax is None:
            ax = plt.gca()

        xy = self.p-alpha

        color = [1, 0, 1, 0.2]
        rect = patches.Rectangle(xy, 2*alpha, 2*alpha, label='D', fill=True, color=color, zorder=2)
        ax.add_patch(rect)


class IllustrativeBetaMapping(ConstraintMapping):
    # A trivial example for plotting
    def __init__(self, env, safety_layer):
        super().__init__(env, safety_layer)

    def get_constraint_boundaries(self, observation, c, augment_action_space=True):
        c = np.array([-1, -1])
        g = np.array([[1, 1],
                      [-1, -1]])

        if augment_action_space:
            c_action_space, g_action_space = self.get_action_space_constraints()
            c = np.concatenate((c, c_action_space))
            g = np.concatenate((g, g_action_space))

            return c, g


# TODO see how this inheritance works
# TODO write this class so that we can use this shit with DDPG



def get_safety_layer(env, fname, train=False, epochs=25):
    from dev.safety_layers.augmented import AugmentedMultipleSafetySecondary
    # TestEnv were 'safety_layer.p'  or  'safety_layer_1.p'
    if train:
        safety_layer = AugmentedMultipleSafetySecondary(env)
        safety_layer._config.epochs = epochs
        safety_layer.train()
        safety_layer.save(fname=fname)
    else:
        safety_layer = AugmentedMultipleSafetySecondary.load(torch.load(fname))

    assert env.action_space.shape == safety_layer._env.action_space.shape
    assert env.observation_space.shape == safety_layer._env.observation_space.shape
    return safety_layer

def ballnd_test():
    from safe_explorer.env.ballnd import BallND2
    env = BallND2()
    p = np.zeros(env.action_space.shape)
    obs = env.reset()
    c = env.get_constraint_values()

    safety_layer = get_safety_layer(env, fname='safety_layer_ballnd.p', train=False, epochs=250)
    mapping = ConstraintMapping(env, safety_layer, p)

    n_samples = 50

    for j in range(n_samples):
        beta_action = np.random.random(env.action_space.shape)
        scaled_action = mapping.map_beta_to_safe(beta_action, obs, c, plot=True, verbose=True)
        mapping.check_constraint_satisfaction(scaled_action, c, obs=obs)
        obs, reward, done, info = env.step(scaled_action)
        c_next = env.get_constraint_values()
        pred_error = mapping.check_constraint_prediction_error(scaled_action, c, c_next, obs=obs)
        safe_action = mapping.safety_layer.get_safe_action(obs, beta_action, c)
        pred_error_2 = mapping.check_constraint_prediction_error(safe_action, c, c_next, obs=obs)
        print('Prediction error: {}, with safety layer projection: {}'.format(pred_error, pred_error_2))
        c = c_next

def ballnd_test_new():
    from safe_explorer.env.ballnd import BallND2
    env = BallND2()
    p = np.zeros(env.action_space.shape)

    safety_layer = get_safety_layer(env, fname='safety_layer_ballnd.p', train=False, epochs=100)
    mapping = ConstraintMapping(env, safety_layer)

    n_samples = 50

    obs = env.reset()
    c = env.get_constraint_values()
    for j in range(n_samples):
        plot = (c > -0.1).any() # Only plot if we are close to the boundary
        beta_action = np.random.rand(*env.action_space.shape)
        if plot:
            env.render()
        scaled_action = mapping.map_beta_to_safe(obs, beta_action, c, plot=plot, verbose=True)
        obs, reward, done, info = env.step(scaled_action)
        c = env.get_constraint_values()

def test_test():
    from envs.test_env import TestEnv

    env = TestEnv()

    beta_action = np.array([0.2, 0.7])
    # p = np.array([-1, 0])

    beta_points = [np.array([0.1, 0.9]), np.array([0.75, 0.4])]

    safety_layer = get_safety_layer(env, 'safety_layer_1.p', train=False, epochs=25)
    mapping = IllustrativeBetaMapping(env, safety_layer)
    obs = env.reset()

    for j in range(4):
        c = env.get_constraint_values()
        beta_action = beta_points[j] if j<len(beta_points) else np.random.rand(2)

        fig_name = 'images/mapping_' + str(round(beta_action[0],2)) + '_' + str(round(beta_action[1],2)) + '.png'
        scaled_action = mapping.map_beta_to_safe(obs, beta_action, c, plot=True, verbose=True, save_fig_name=fig_name)
        obs, reward, done, info = env.step(scaled_action)

if __name__ == '__main__':
    # ballnd_test_new()
    test_test()
