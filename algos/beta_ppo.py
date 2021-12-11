from dev.algos.policy_gradient.stochastic import PPO
from beta_constraints.mapping.beta_mapping import ConstraintMapping
import torch
from gym import Env

class BetaConstrainedPPO(PPO):
    def __init__(self, env, safety_layer, actor_critic_kwargs=None, seed=None,
                 steps_per_epoch=4000, epochs=50,
                 gamma=0.99, clip_ratio=0.2, pi_lr=3e-4,
                 vf_lr=1e-3, lam=0.97,
                 logger_kwargs=None, progress_logger_kwargs=None, clip_actions=False, min_epoch_len=None, **kwargs):

        if 'policy_distribution' in actor_critic_kwargs and actor_critic_kwargs['policy_distribution'] != 'beta':
            print('Changing ActorCritic policy_distribution from {} to beta')
        actor_critic_kwargs['policy_distribution'] = 'beta'

        super().__init__(env, actor_critic_kwargs=actor_critic_kwargs, seed=seed,
                         steps_per_epoch=steps_per_epoch, epochs=epochs,
                         gamma=gamma, clip_ratio=clip_ratio, pi_lr=pi_lr,
                         vf_lr=vf_lr, lam=lam, logger_kwargs=logger_kwargs,
                         progress_logger_kwargs=progress_logger_kwargs,
                         clip_actions=clip_actions, min_epoch_len=min_epoch_len, **kwargs)

        assert isinstance(env, Env)

        self.mapping = ConstraintMapping(env, safety_layer)

    def _get_action(self, observation):
        beta_action, value, log_prob = self.actor_critic.step(torch.as_tensor(observation, dtype=torch.float32), )
        info = dict(old_action=beta_action, old_log_prob=log_prob)

        c = self.env.get_constraint_values()
        c_region_action = self.mapping.map_beta_to_safe(observation, beta_action, c)
        return c_region_action, value, log_prob, info