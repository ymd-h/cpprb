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

        self.assertEqual(np.arange(size)[_a[0]],_a[0])

class TestFeatureHighDimensionalObs(unittest.TestCase):
    def test_RGB_screen_obs(self):
        size = 256
        obs_shape = (84,84,3)
        act_dim = 3

        rb = create_buffer(size,obs_shape=obs_shape,act_dim=act_dim,
                           prioritized = True,
                           is_discrete_action = True)

        obs = np.ones(obs_shape,dtype=np.double)
        act = 2
        rew = 0.5
        next_obs = np.zeros_like(obs)
        done = 0

        rb.add(obs,act,rew,next_obs,done)

        _o = rb._encode_sample(np.array((0)))["obs"]
        _no = rb._encode_sample(np.array((0)))["next_obs"]

        self.assertEqual(obs_shape,_o[0].shape)
        np.testing.assert_allclose(obs,_o[0])

        self.assertEqual(obs_shape,_no[0].shape)
        np.testing.assert_allclose(next_obs,_no[0])

    def test_BatchSampling(self):
        size = 256
        obs_shape = (84,84,3)
        act_dim = 3
        batch_size = 64

        rb = create_buffer(size,obs_shape=obs_shape,act_dim=act_dim,
                           prioritized = True,
                           Nstep = True,
                           is_discrete_action = True)

        obs = np.ones(obs_shape,dtype=np.double)
        act = 2
        rew = 0.5
        next_obs = np.zeros_like(obs)
        done = 0

        rb.add(obs,act,rew,next_obs,done)

        rb.sample(batch_size)

if __name__ == '__main__':
    unittest.main()
