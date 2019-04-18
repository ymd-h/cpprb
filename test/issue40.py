import numpy as np

from cpprb import ReplayBuffer

buffer_size = 256
obs_dim = 3
act_dim = 1
rb = ReplayBuffer(buffer_size,obs_dim,act_dim)

obs = np.ones(shape=(obs_dim))
act = np.ones(shape=(act_dim))
rew = 0
next_obs = np.ones(shape=(obs_dim))
done = 0

for i in range(500):
    rb.add(obs,act,rew,next_obs,done)


batch_size = 32
sample = rb.sample(batch_size)
