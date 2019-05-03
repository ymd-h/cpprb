import numpy as np
import unittest

from cpprb import create_buffer


class TestFeatureDiscreteAction(unittest.TestCase):
    def test_discrete(self):
        size = 256
        obs_dim = 15
        act_dim = 7

        rb = create_buffer(size,obs_dim,act_dim,is_discrete_action = True)

        obs = np.arange(obs_dim)
        act = 3
        rew = 1.0
        next_obs = np.zeros(obs_dim,dtype=np.double)
        done = 0

        rb.add(obs,act,rew,next_obs,done)

        _a = rb._encode_sample(np.array((0),dtype=np.int))["act"]

        self.assertIs(np.dtype('int64'),_a.dtype)

        self.assertEqual(rb._encode_sample(np.array((0),dtype=np.int))["obs"][_a[0]],
                         _a[0])

if __name__ == '__main__':
    unittest.main()
