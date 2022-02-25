from safe_explorer.env.ballnd import BallND2
from safe_explorer.env.spaceship import Spaceship2
from experiments.env_trainer import EnvTrainer
from algos.beta_ppo import BetaConstrainedPPO
from dev.safety_layers.safe_microgrid import MPIMultipleSafetyLayer
from torch import nn

class BallNDTrainer(EnvTrainer):
    def __init__(self, filename_suffix=None, set_seeds=True):
        super().__init__(BallND2(), config_name='ballnd_standard_safety_layer', filename_suffix=filename_suffix,
                         set_seeds=set_seeds)

class SpaceshipTrainer(EnvTrainer):
    def __init__(self, filename_suffix=None, set_seeds=True):
        super().__init__(Spaceship2(), config_name='spaceship_standard_safety_layer', filename_suffix=filename_suffix,
                         set_seeds=set_seeds)


if __name__=='__main__':

    test = True
    profile = False
    suffix = 'l2_gaussian_3'


    if test:
        episodes_per_epoch = 2
        epochs = 500
        do_mpi_fork = False
        n_cpu = episodes_per_epoch
        max_episode_len = 450
        min_epoch_len = 900

    else:
        episodes_per_epoch = 2
        epochs = 250
        do_mpi_fork = True
        n_cpu = episodes_per_epoch
        max_episode_len = 300
        min_epoch_len = 300


    ppo_init_kwargs = dict(algo='ppo',
        epochs=epochs,
        steps_per_epoch=min_epoch_len * episodes_per_epoch,
        actor_critic_kwargs=dict(activation=nn.Tanh, hidden_sizes=(64, 64), policy_distribution = 'beta'),
        gamma=0.99,
        clip_ratio=0.2,
        lam=0.9,
        target_kl=0.01,
        clip_actions=True,
        min_epoch_len=min_epoch_len)

    ppo_run_kwargs = dict(max_episode_len=max_episode_len)
    safety_layer_kwargs = dict(load=True,
                               fname='pickle/spaceship_standard_safety_layer_beta_mapping.p'
                               )

    trainer = SpaceshipTrainer(filename_suffix=suffix, set_seeds=False)
    if not profile:
        pickle_dict = trainer.train(BetaConstrainedPPO, MPIMultipleSafetyLayer, safety_layer_kwargs, ppo_init_kwargs, ppo_run_kwargs)
    else:
        import cProfile
        cProfile.run('trainer.train(SafePPO, MultipleSafetyLayer, safety_layer_kwargs, ppo_init_kwargs, ppo_run_kwargs)', filename='BetaSafePPOProfile.profile')
