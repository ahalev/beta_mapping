import dill as pickle
from experiments.ballnd_ppo import BallNDTrainer

fname ='pickle/ballnd_standard_safety_layer.p'
f = open(fname, 'rb')
experiment = pickle.load(f)
f.close()
ppo, safety_layer = experiment['ppo'], experiment['safety_layer']

# BallNDTrainer.render_policy(ppo)

from halev_dev.algos.actor_critic import BetaActor
from torch import nn
observation_dim = 2
action_dim = 3
hidden_sizes=(64,64)
b = BetaActor(observation_dim, action_dim, hidden_sizes, activation=nn.Tanh)
print('initiated')