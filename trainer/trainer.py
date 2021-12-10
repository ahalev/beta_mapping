import sys
from datetime import datetime
from torch import nn
from dev.algos.policy_gradient.stochastic.PPO import SafePPO
from safe_explorer.ddpg.ddpg import DDPG
from dev.safety_layers import MultipleSafetyLayer
from dev.safety_layers import AugmentedMultipleSafetySecondary
from dev.envs.safe import SafeExpMicrogridEnv
from dev.experiment.trainer.env_trainer import EnvTrainer

test = True
profile = False
suffix = 'augmented_secondary_layer'

env_cls = SafeExpMicrogridEnv
safety_layer_cls = AugmentedMultipleSafetySecondary
ppo_cls = SafePPO

if test:
    episodes_per_epoch = 2
    epochs = 10
    safety_layer_trainer_epochs = 10
    mpi_cpus = 1  # 1: no mpi fork, >1: mpi fork
    max_episode_len = 450
    min_epoch_len = 900
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    suffix += "_test_"
    suffix += current_time

else:
    episodes_per_epoch = 32
    epochs = 1000
    safety_layer_trainer_epochs = 1000
    # do_mpi_fork = True
    mpi_cpus = 16  # mpi fork
    max_episode_len = 5840
    min_epoch_len = 5840

env_init_args = 0,  # microgrid number

env_init_kwargs = dict(
    # microgrid_num=1,
    # trajectory_len=None,
    max_episode_len=max_episode_len,
    only_inequality_constr=False,
    periodic_time=True,
    genset_implementation='standard',
    reward_shaping_func='scale_mgrid_load'
)

ppo_init_kwargs = dict(algo='ppo',
                       epochs=epochs,
                       steps_per_epoch=min_epoch_len * episodes_per_epoch,
                       actor_critic_kwargs=dict(activation=nn.Tanh, hidden_sizes=(64, 64),
                                                policy_distribution='gaussian'),
                       gamma=0.99,
                       clip_ratio=0.2,
                       lam=0.9,
                       target_kl=0.01,
                       clip_actions=True,
                       min_epoch_len=min_epoch_len)

ppo_run_kwargs = dict(max_episode_len=max_episode_len)
safety_layer_trainer_config = dict(max_episode_length=max_episode_len,
                                   epochs=safety_layer_trainer_epochs,
                                   steps_per_epoch= episodes_per_epoch*min_epoch_len
                                   )
safety_layer_kwargs = dict(load=False,
                           fname='pickle/SafeExpMicrogridEnv_augmented_secondary_layer.p',
                           trainer_config=safety_layer_trainer_config
                           )

trainer_config = dict(seed=0)
trainer = EnvTrainer(env_cls, env_init_args, env_init_kwargs, trainer_config, filename_suffix=suffix, set_seeds=True)
if not profile:
    pickle_dict = trainer.train(safety_layer_cls, ppo_cls, safety_layer_kwargs, ppo_init_kwargs, ppo_run_kwargs,
                                mpi_cpus=mpi_cpus)
else:
    import cProfile

    cProfile.run('trainer.train(SafePPO, MultipleSafetyLayer, safety_layer_kwargs, ppo_init_kwargs, ppo_run_kwargs)',
                 filename='BetaSafePPOProfile.profile')

sys.exit(0)
