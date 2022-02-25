from safe_explorer.core.config import Config
import torch
import numpy as np
from dev.utils.mpi_tools import proc_id
from dev.algos.policy_gradient.stochastic.PPO import PPO, SafePPO
from dev.safety_layers.safe_microgrid import MPIMultipleSafetyLayer, StreamlinedSafetyLayer
from dev.safety_layers.augmented import AugmentedMultipleSafetyLayer, AugmentedMultipleSafetySecondary
from dev.safety_layers.second_order import SecondOrderMultipleSafetyLayer
from dev.utils.utils import colorize
from dev.utils.writer import TensorBoard
from pathlib import Path
import dill as pickle
from torch import nn
from matplotlib import pyplot as plt
import inspect
from algos.beta_ppo import BetaConstrainedPPO

class EnvTrainer:
    def __init__(self, env, set_seeds=True, config_name='', filename_suffix=None):
        self.env=env
        self._config = Config.get().main.trainer
        self._set_seeds(set_seeds)
        self.pickle_dict = dict()
        self.config_name = config_name
        self.pickle_name, self.logger_kwargs = self.set_filenames(suffix=filename_suffix)

    def _set_seeds(self, set_seeds):
        if set_seeds:
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
        logger_kwargs = dict(output_dir=str(logger_file))
        return pickle_name, logger_kwargs

    def train(self, ppo_cls, safety_layer_cls, safety_layer_kwargs, ppo_init_kwargs, ppo_run_kwargs):
        writer = self.init_tb_writer()
        safety_layer = self.get_safety_layer(safety_layer_cls, **safety_layer_kwargs)
        self.pickler()
        _ = self.run_ppo(ppo_cls, safety_layer, ppo_init_kwargs, ppo_run_kwargs)
        self.pickler()
        return self.pickle_dict

    def init_tb_writer(self):
        return TensorBoard.get_writer_dirspec('tensorboard', new_writer=True, comment=self.config_name)

    def get_safety_layer(self, safety_layer_cls, load=False, fname=None, **layer_kwargs):
        """
        Return a trained safety layer
        Make sure you append pickle_dict in here
        :param: load, bool. If True, loads a safety_layer stored in the file fname.
        :param: fname, str or None. If load, file from which safety layer loaded. Otherwise, not used.

        :return:
        """
        if load:
            safety_layer = self.load_safety_layer(fname)
            # safety_layer.constrain_method = 'multiplier'
        else:
            try:
                safety_layer = safety_layer_cls(self.env, constrain_method='cvxpy_param')
            except TypeError:
                safety_layer = safety_layer_cls(self.env, **layer_kwargs)
            safety_layer.train()

        self.pickle_dict['safety_layer'] = safety_layer
        return safety_layer

    def load_safety_layer(self, fname):
        import dill as pickle
        f = open(fname, 'rb')

        try:
            d = pickle.load(f)
        except pickle.UnpicklingError as e: # Check if a /tmp/experiments/<>.txt file was erased and replace it
            import os
            f.close()
            path = Path(e.args[0].filename).parent
            os.makedirs(path)
            f_temp = open(e.args[0].filename, 'w')
            f_temp.close()
            f = open(fname, 'rb')
            d = pickle.load(f)

        f.close()
        if isinstance(d, MPIMultipleSafetyLayer):
            return d
        elif 'safety_layer' in d.keys():
            return d['safety_layer']
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
        init_kwargs['logger_kwargs'] = self.logger_kwargs
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
            print('Nothing in pickle_dict, exiting without pickling')
            return

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
