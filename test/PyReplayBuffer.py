import numpy as np
import unittest
from ymd_K import ReplayBuffer

class TestPyReplayBuffer(unittest.TestCase):
    """=== PyReplayBuffer.py ==="""

    obs_dim = 3
    act_dim = 5

    N_step = 100
    N_buffer_size = 15
    N_sample = 5

    @classmethod
    def setUpClass(cls):
        cls.rb = ReplayBuffer.PyReplayBuffer(cls.N_buffer_size,
                                             cls.obs_dim,
                                             cls.act_dim)
        for i in range(cls.N_step):
            cls.rb.add(np.zeros(shape=cls.obs_dim),
                       np.ones(shape=cls.act_dim),
                       0.5*i,
                       np.ones(shape=cls.obs_dim)*i,
                       0 if i is not cls.N_step - 1 else 1)
        cls.s = cls.rb.sample(cls.N_sample)

    def _check_ndarray(self,array,ndim,shape):
        self.assertEqual(ndim,array.ndim)
        self.assertEqual(shape,array.shape)

    def test_obs(self):
        self._check_ndarray(self.s['obs'],2,(self.N_sample,self.obs_dim))

    def test_act(self):
        self._check_ndarray(self.s['act'],2,(self.N_sample,self.act_dim))

    def test_rew(self):
        self._check_ndarray(self.s['rew'],1,(self.N_sample,))

    def test_next_obs(self):
        self._check_ndarray(self.s['next_obs'],2,(self.N_sample,self.obs_dim))

        for i in range(self.N_sample):
            self.assertGreaterEqual(self.s['next_obs'][i,0],
                                    self.N_step - self.N_buffer_size)
            self.assertLess(self.s['next_obs'][i,0],self.N_step)

            for j in range(1,self.obs_dim):
                self.assertAlmostEqual(self.s['next_obs'][i,0],
                                       self.s['next_obs'][i,j])

    def test_done(self):
        self._check_ndarray(self.s['done'],1,(self.N_sample,))

if __name__ == '__main__':
    unittest.main()
