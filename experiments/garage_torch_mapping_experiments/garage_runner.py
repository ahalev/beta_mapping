from beta_constraints.mapping import ConstraintMapping, TorchConstraintMapping
from beta_constraints.mapping.beta_mapping import get_safety_layer
from algos.ddpg.actor.garage_actor.garage_actor import DeterministicMLPPolicy, DeterministicMappingPolicy
from dev.envs.wrapper.save_wrapper import SaveWrapper
from torch import nn
from safe_explorer.env.ballnd import BallND2
from safe_explorer.env.spaceship import Spaceship
from envs.gym_env.gym_env import GymEnv
from garage.torch.algos.ddpg import DDPG
from garage.trainer import Trainer
from garage.torch.q_functions import ContinuousMLPQFunction
from garage.replay_buffer import PathBuffer
from garage.sampler import LocalSampler
from garage import wrap_experiment
from garage.np.exploration_policies import AddOrnsteinUhlenbeckNoise
env = GymEnv(SaveWrapper(BallND2(n=2)))
env_2 = GymEnv(SaveWrapper(env_cls=Spaceship, env_args=()))

@wrap_experiment
def beta_experiment(ctxt=None):
    safety_layer = get_safety_layer(env, fname='safety_augmented_layer_ballnd.p', train=False, epochs=50)
    torch_mapping = TorchConstraintMapping(env, safety_layer)
    numpy_mapping = ConstraintMapping(env, safety_layer)
    comparison_policy = DeterministicMLPPolicy(env.spec, hidden_sizes=(128,128), output_nonlinearity=nn.Sigmoid)
    # comparison_policy.load_state_dict(mapping_policy.state_dict())

    hyperparams = dict(n_epochs=100,
                       steps_per_epoch=10,
                       sampler_batch_size=1500,
                       max_episode_len = 300,
                       lr=1e-4,
                       discount=0.99,
                       min_buffer_size=int(1e4),
                       n_train_steps=125,
                       target_update_freq=2,
                       buffer_batch_size=32,
                       max_epsilon=1.0,
                       min_epsilon=0.01,
                       decay_ratio=0.1,
                       buffer_size=int(1e4),
                       # hidden_sizes=(512, ),
                       hidden_sizes = (32,32),
                       hidden_channels=(32, 64, 64),
                       kernel_sizes=(8, 4, 3),
                       strides=(4, 2, 1),
                       clip_gradient=10)

    env_spec = env.spec
    mapping_policy = DeterministicMappingPolicy(env, safety_layer)
    safety_layer_writer = mapping_policy.mapping.safety_layer._writer
    del mapping_policy.mapping.safety_layer._writer
    exploration_policy = AddOrnsteinUhlenbeckNoise(env.spec, mapping_policy, sigma=0.2)
    sampler = LocalSampler(agents=exploration_policy, envs = env, max_episode_length=hyperparams['max_episode_len'])
    qf = ContinuousMLPQFunction(env_spec, hidden_sizes=hyperparams['hidden_sizes'])
    replay_buffer = PathBuffer(hyperparams['buffer_size'])
    ddpg = DDPG(env_spec=env_spec, policy=mapping_policy, qf=qf, replay_buffer=replay_buffer, sampler=sampler, exploration_policy=exploration_policy,
                steps_per_epoch=hyperparams['steps_per_epoch'])

    runner = Trainer(ctxt)
    runner.setup(ddpg, env)
    runner.train(n_epochs=hyperparams['n_epochs'],
                  batch_size=hyperparams['sampler_batch_size'])

# beta_experiment()

import cProfile

cProfile.run('beta_experiment()',
             filename='beta_ddpg_profile.profile')
