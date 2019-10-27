import numpy as np
import unittest

from cpprb import ReplayBuffer

class TestMultiRew(unittest.TestCase):
    def test(self):
        buffer_size = 256
        obs_dim = 3
        act_dim = 1
        rew_dim = 2
        rb = ReplayBuffer(buffer_size,
                          {"obs": {"shape": obs_dim},
                           "act": {"shape": act_dim},
                           "rew": {"shape": rew_dim},
                           "next_obs": {"shape": obs_dim},
                           "done": {}})

        obs = np.ones(shape=(obs_dim))
        act = np.ones(shape=(act_dim))
        rew = (0,1)
        next_obs = np.ones(shape=(obs_dim))
        done = 0

        for i in range(500):
            rb.add(obs=obs,act=act,rew=rew,next_obs=next_obs,done=done)


        batch_size = 32
        sample = rb.sample(batch_size)

        self.assertEqual(0,sample["rew"][0,0])
        self.assertEqual(1,sample["rew"][0,1])

if __name__ == '__main__':
    unittest.main()
