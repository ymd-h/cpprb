import unittest

import numpy as np

from cpprb import ReplayBuffer
from cpprb.experimental import ReplayBuffer as expReplayBuffer

class TestExperimentalReplayBuffer(unittest.TestCase):
    def test_buffer(self):

        buffer_size = 256
        obs_shape = (15,15)
        act_dim = 5

        N = 512

        rb = ReplayBuffer(buffer_size,obs_shape=obs_shape,act_dim=act_dim)
        erb = expReplayBuffer(buffer_size,{"obs":{"shape": obs_shape},
                                           "act":{"shape": act_dim},
                                           "rew":{},
                                           "next_obs":{"shape": obs_shape},
                                           "done":{}})

        for i in range(N):
            obs = np.full(obs_shape,i,dtype=np.double)
            act = np.full(act_dim,i,dtype=np.double)
            rew = i
            next_obs = obs + 1
            done = 0

            rb.add(obs,act,rew,next_obs,done)
            erb.add(obs=obs,act=act,rew=rew,next_obs=next_obs,done=done)


        s = rb._encode_sample(range(N))
        es = erb._encode_sample(range(N))

        np.testing.assert_allclose(s["obs"],es["obs"])
        np.testing.assert_allclose(s["act"],es["act"])
        np.testing.assert_allclose(s["rew"],es["rew"])
        np.testing.assert_allclose(s["next_obs"],es["next_obs"])
        np.testing.assert_allclose(s["done"],es["done"])

if __name__ == '__main__':
    unittest.main()
