from ymd_K import ReplayBuffer
import numpy as np

obs_dim = 3
act_dim = 5

N_step = 100

rb = ReplayBuffer.PyReplayBuffer(15,obs_dim,act_dim)


for i in range(N_step):
    rb.add(np.zeros(shape=obs_dim),
           np.ones(shape=act_dim),
           0.5*i,
           np.ones(shape=obs_dim)*i,
           0 if i is not N_step - 1 else 1)

s = rb.sample(5)
print(s['obs'])
print(s['act'])
print(s['rew'])
print(s['next_obs'])
print(s['done'])
