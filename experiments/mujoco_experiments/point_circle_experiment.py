from safe_explorer.env.ballnd import BallND2
from safe_explorer.core.config import Config
import torch
import numpy as np
from halev_dev.utils.mpi_tools import proc_id
from halev_dev.algos.PPO import PPO, SafePPO
from halev_dev.algos.safe_microgrid import MultipleSafetyLayer, AugmentedMultipleSafetyLayer, \
    AugmentedMultipleSafetySecondary, SecondOrderMultipleSafetyLayer
from halev_dev.utils.utils import colorize
from halev_dev.utils.writer import TensorBoard
from pathlib import Path
import dill as pickle
from torch import nn
from matplotlib import pyplot as plt
import inspect
from envs.cpo_envs_modified.mujoco_safe.point_env_safe import SafePointEnv
from algos.beta_ppo import BetaConstrainedPPO

class EnvTrainer:
    def __init__(self, env, config_name='', filename_suffix=None):
        self.env=env
        self._config = Config.get().main.trainer
        self._set_seeds()
        self.pickle_dict = dict()
        self.config_name = config_name
        self.pickle_name, self.logger_kwargs = self.set_filenames(suffix=filename_suffix)

    def _set_seeds(self):
        torch.manual_seed(self._config.seed+proc_id())
        np.random.seed(self._config.seed+proc_id())

    def set_filenames(self, suffix=None):
        if self.config_name == '':
            self.config_name = self.env.__str__()
        if suffix is not None:
            self.config_name += '_' + suffix

        pickle_name = 'pickle/' + self.config_name + '.p'
        logger_name = 'logs/' + self.config_name
        logger_file = Path(__file__).parent / logger_name
        logger_kwargs = dict(output_dir=logger_file)
        return pickle_name, logger_kwargs

    def train(self, ppo_cls, safety_layer_cls, safety_layer_kwargs, ppo_init_kwargs, ppo_run_kwargs):
        writer = self.init_tb_writer()
        safety_layer = self.get_safety_layer(safety_layer_cls, **safety_layer_kwargs)
        self.pickler()
        _ = self.run_ppo(ppo_cls, safety_layer, ppo_init_kwargs, ppo_run_kwargs)
        self.pickler()
        return self.pickle_dict

    def init_tb_writer(self):
        return TensorBoard.get_writer_dirspec('tensorboard_new', new_writer=True, comment=self.config_name)

    def get_safety_layer(self, safety_layer_cls, load=False, fname=None):
        """
        Return a trained safety layer
        Make sure you append pickle_dict in here
        :param: load, bool. If True, loads a safety_layer stored in the file fname.
        :param: fname, str or None. If load, file from which safety layer loaded. Otherwise, not used.

        :return:
        """
        if load:
            safety_layer =  self.load_safety_layer(fname)
            self.pickle_dict['safety_layer'] = safety_layer
            print('Using saved safety layer with config:')
            safety_layer._config.pprint()
            return safety_layer

        else:
            safety_layer = safety_layer_cls(self.env, constrain_method='cvxpy_param')
            safety_layer.train()
            self.pickle_dict['safety_layer'] = safety_layer
            return safety_layer

    def load_safety_layer(self, fname):
        import dill as pickle
        f = open(fname, 'rb')
        d = pickle.load(f)
        f.close()
        if isinstance(d, MultipleSafetyLayer):
            return d
        elif 'safety_layer' in d.keys():
            return d['safety_layer']
        elif 'ppo' in d.keys() and hasattr(d['ppo'], 'safety_layer'):
            print('Using safety_layer stored in SafePPO')
            return d['ppo'].safety_layer
        else:
            raise NameError('Unable to find safety_layer in dict with keys {}'.format(d.keys()))

    def run_ppo(self, algorithm, safety_layer, init_kwargs, run_kwargs):
        """
        Return a trained PPO
        Also make sure you append to pickle dict in here
        :param env:
        :param safety_layer:
        :param init_kwargs:
        :param run_kwargs:
        :return:
        """
        print('Initiating algorithm {}'.format(algorithm))

        if 'safety_layer' not in inspect.signature(algorithm).parameters:
            if safety_layer is not None:
                print('Warning, you passed a safety layer but algorithm {} does not require a safety layer. '
                      'Ignoring safety layer'.format(algorithm))
            ppo = algorithm(self.env, **init_kwargs)
        else:
            ppo = algorithm(self.env, safety_layer, **init_kwargs)
        ppo.run(**run_kwargs)
        self.pickle_dict['ppo'] = ppo
        return ppo

    def pickler(self):
        if len(self.pickle_dict) == 0:
            raise ValueError('Nothing in pickle_dict')

        if 'safety_layer' in self.pickle_dict:
            self.pickle_dict['safety_layer'].reset()
        elif hasattr(self.pickle_dict['ppo'], 'safety_layer'):
            self.pickle_dict['ppo'].safety_layer.reset()
        elif hasattr(self.pickle_dict['ppo'], 'mapping'):
            self.pickle_dict['ppo'].mapping.safety_layer.reset()

        f = open(self.pickle_name, 'wb')
        pickle.dump(self.pickle_dict, f)
        f.close()

        if proc_id() == 0:
            print(colorize('Successfully pickled pickle_dict with elements {} as {}'.format(self.pickle_dict.keys(), self.pickle_name), 'green',
                           bold=True))

    @staticmethod
    def render_policy(ppo, max_steps = 1000):
        actor_critic = ppo.actor_critic
        env = ppo.env
        layer = ppo.safety_layer

        obs = env.reset()
        done = False

        for _ in range(max_steps):
            if done:
                break
            action, _, _ = actor_critic.step(obs)
            c = env.get_constraint_values()
            action = layer.get_safe_action(obs, action, c)
            obs, reward, done, info = env.step(action)
            plt.close('all')
            env.render()

if __name__=='__main__':

    test = False
    profile = False
    suffix= 'gaussian_l2_projection_radial_action_second_order_layer'


    if test:
        episodes_per_epoch = 4
        epochs = 5
        do_mpi_fork = False
        n_cpu = episodes_per_epoch
        max_episode_len = 300
        min_epoch_len = 300

    else:
        episodes_per_epoch = 4
        epochs = 500
        do_mpi_fork = True
        n_cpu = 2
        max_episode_len = 300
        min_epoch_len = 300

    ppo_init_kwargs = dict(algo='ppo',
        epochs=epochs,
        steps_per_epoch=min_epoch_len * episodes_per_epoch,
        actor_critic_kwargs=dict(activation=nn.Tanh, hidden_sizes=(64, 64), policy_distribution = 'gaussian'),
        gamma=0.99,
        clip_ratio=0.2,
        lam=0.9,
        target_kl=0.01,
        clip_actions=True,
        min_epoch_len=min_epoch_len)

    with_buffer=True # TODO when this is done, train with a buffer, then train w/o termination in SafetyLayer (but with in PPO)
    terminate_on_violation = True

    if with_buffer:
        constraint_buffer = 0.1
        suffix += '_with_buffer'
    else:
        constraint_buffer = 0.0
        suffix += '_no_buffer'
    env = SafePointEnv(terminate_on_violation=terminate_on_violation, constraint_buffer=constraint_buffer,
                       cartesian_action=False)

    ppo_run_kwargs = dict(max_episode_len=max_episode_len)
    safety_layer_kwargs = dict(load=True,
                               fname='pickle/point_circle_second_order_safety_layer_gaussian_l2_projection_radial_action_second_order_layer_with_buffer.p')

    trainer = EnvTrainer(env, config_name = 'point_circle_second_order_safety_layer', filename_suffix=suffix)
    if not profile:
        pickle_dict = trainer.train(SafePPO, SecondOrderMultipleSafetyLayer, safety_layer_kwargs, ppo_init_kwargs, ppo_run_kwargs)
    else:
        import cProfile
        cProfile.run('trainer.train(SafePPO, MultipleSafetyLayer, safety_layer_kwargs, ppo_init_kwargs, ppo_run_kwargs)', filename='BetaSafePPOProfile.profile')
