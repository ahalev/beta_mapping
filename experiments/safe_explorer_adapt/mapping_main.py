from functional import seq
import numpy as np
import torch

from dev.safety_layers import AugmentedMultipleSafetySecondary, MPIMultipleSafetyLayer, AugmentedMultipleSafetyLayer
from safe_explorer.core.config import Config
from safe_explorer.env.ballnd import BallND2
from safe_explorer.env.spaceship import Spaceship
from safe_explorer.ddpg.critic import Critic
from safe_explorer.ddpg.ddpg import DDPG
from algos.ddpg.actor.safe_explorer_actor.ddpg_map_actor import SafeExplorerActor


class MappingTrainer:
    def __init__(self):
        self._config = Config.get().main.trainer
        self._set_seeds()

    def _set_seeds(self):
        torch.manual_seed(self._config.seed)
        np.random.seed(self._config.seed)

    def _print_ascii_art(self):
        print(
            """
              _________       _____        ___________              .__                              
             /   _____/____ _/ ____\____   \_   _____/__  _________ |  |   ___________   ___________ 
             \_____  \\__  \\   __\/ __ \   |    __)_\  \/  /\____ \|  |  /  _ \_  __ \_/ __ \_  __ \\
             /        \/ __ \|  | \  ___/   |        \>    < |  |_> >  |_(  <_> )  | \/\  ___/|  | \/
            /_______  (____  /__|  \___  > /_______  /__/\_ \|   __/|____/\____/|__|    \___  >__|   
                    \/     \/          \/          \/      \/|__|                           \/    
            """)

    def train(self):
        self._print_ascii_art()
        print("============================================================")
        print("Initialized SafeExplorer with config:")
        print("------------------------------------------------------------")
        Config.get().pprint()
        print("============================================================")

        env = BallND2(n=2) if self._config.task == "ballnd" else Spaceship()

        if self._config.use_safety_layer:
            safety_layer = AugmentedMultipleSafetySecondary(env, config=dict(epochs=250, constraint_model_layers = [20, 20]))
            safety_layer.train()
        try:
            observation_dim = (seq(env.observation_space.spaces.values())
                               .map(lambda x: x.shape[0])
                               .sum())
        except AttributeError:
            observation_dim = env.observation_space.shape[0]

        actor = SafeExplorerActor(env, safety_layer, observation_dim, env.action_space.shape[0])
        del actor.mapping.safety_layer._writer

        critic = Critic(observation_dim, env.action_space.shape[0])

        safe_action_func = safety_layer.get_safe_action if safety_layer else None
        safe_action_func = None
        ddpg = DDPG(env, actor, critic, safe_action_func)

        ddpg.train()


if __name__ == '__main__':
    MappingTrainer().train()