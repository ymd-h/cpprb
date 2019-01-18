import numpy as np
from ymd_K import ReplayBuffer

print("=== PyReplayBuffer.py ===")

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
print("obs: {}".format(s['obs']))
print("act: {}".format(s['act']))
print("rew: {}".format(s['rew']))
print("next_obs: {}".format(s['next_obs']))
print("done: {}".format(s['done']))

print("obs.shape: {}".format(s['obs'].shape))
print("done.shape: {}".format(s['done'].shape))
print("done.ndim: {}".format(s['done'].ndim))
print("done.dtype: {}".format(s['done'].dtype))
print("done.strides: {}".format(s['done'].strides))
print("done.base: {}".format(s['done'].base))
print("done.data: {}".format(s['done'].data))
