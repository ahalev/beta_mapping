import torch.nn as nn
from algos.ddpg.actor.actor_critic import mlp
from algos.ddpg.actor.utils import apply_along_axis
from beta_constraints.mapping.torch_mapping import TorchConstraintMapping
from safe_explorer.core.config import Config


class MappingActor(nn.Module):
    def __init__(self, env, safety_layer, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        pi_sizes = [obs_dim] + list(hidden_sizes) + [act_dim]
        self.pi = mlp(pi_sizes, activation, output_activation=nn.Sigmoid) # outputs in the unit cube
        self.mapping = TorchConstraintMapping(env, safety_layer)
        self.obs_dim = self.mapping.env.observation_space.shape[0]
        self.action_dim = self.mapping.env.action_space.shape[0]

    def forward(self, obs):
        obs_view = obs.view(-1, self.obs_dim)
        beta_action = self.pi(obs)
        action_view = beta_action.view(-1, self.action_dim)
        safe_region_action = apply_along_axis(self.mapping_function, obs_view, action_view)
        # safe_region_action = self.mapping(obs, beta_action, None)
        return safe_region_action.squeeze()

    def mapping_function(self, observations, actions):
        return self.mapping(observations, actions, None)

class SafeExplorerActor(MappingActor):
    def __init__(self, env, safety_layer, observation_dim, action_dim, activation=nn.ReLU):
        config = Config.get().ddpg.actor

        super().__init__(env,
                            safety_layer,
                            observation_dim,
                            action_dim,
                            config.layers,
                            activation,
                            )