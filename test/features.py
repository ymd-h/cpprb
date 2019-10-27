import numpy as np
import unittest

from cpprb import create_buffer

class TestFeatureHighDimensionalObs(unittest.TestCase):
    def test_RGB_screen_obs(self):
        size = 256
        obs_shape = (84,84,3)
        act_dim = 3

        rb = create_buffer(size,{"obs": {"shape": obs_shape},
                                 "act": {"shape": act_dim},
                                 "rew": {},
                                 "next_obs": {"shape": obs_shape},
                                 "done": {}},
                           prioritized = True)

        obs = np.ones(obs_shape,dtype=np.double)
        act = 2
        rew = 0.5
        next_obs = np.zeros_like(obs)
        done = 0

        rb.add(obs=obs,act=act,rew=rew,next_obs=next_obs,done=done)

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
