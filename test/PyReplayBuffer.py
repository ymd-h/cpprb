import numpy as np
import unittest, time
from ymd_K import ReplayBuffer

class TestPyReplayBuffer(unittest.TestCase):
    """=== PyReplayBuffer.py ==="""

    obs_dim = 3
    act_dim = 1

    N_step = 100
    N_buffer_size = 32
    N_sample = 16

    @classmethod
    def setUpClass(cls):
        cls.rb = ReplayBuffer.PyReplayBuffer(cls.N_buffer_size,
                                             cls.obs_dim,
                                             cls.act_dim)
        for i in range(cls.N_step):
            cls.rb.add(np.ones(shape=(1,cls.obs_dim))*i,
                       np.zeros(shape=(1,cls.act_dim)),
                       np.ones((1)) * 0.5*i,
                       np.ones(shape=(1,cls.obs_dim))*(i+1),
                       np.zeros((1),dtype=np.int32) if i is not cls.N_step - 1 else np.ones((1),dtype=np.int32),
                       1)
        cls.s = cls.rb.sample(cls.N_sample)

    def _check_ndarray(self,array,ndim,shape,name):
        self.assertEqual(ndim,array.ndim)
        self.assertEqual(shape,array.shape)
        print("ER " + name + " {}".format(array))

    def test_obs(self):
        self._check_ndarray(self.s['obs'],min(2,self.obs_dim),
                            (self.N_sample,
                             self.obs_dim) if self.obs_dim > 1 else (self.N_sample,),
                            "obs")

    def test_act(self):
        self._check_ndarray(self.s['act'],min(2,self.act_dim),
                            (self.N_sample,
                             self.act_dim) if self.act_dim > 1 else (self.N_sample,),
                            "act")

    def test_rew(self):
        self._check_ndarray(self.s['rew'],1,(self.N_sample,),"rew")

    def test_next_obs(self):
        self._check_ndarray(self.s['next_obs'],min(2,self.obs_dim),
                            (self.N_sample,
                             self.obs_dim) if self.obs_dim > 1 else (self.obs_dim,),
                            "next_obs")

        for i in range(self.N_sample):
            self.assertGreaterEqual(self.s['next_obs'][i,0],
                                    self.N_step - self.N_buffer_size)
            self.assertLess(self.s['next_obs'][i,0],self.N_step+1)

            for j in range(1,self.obs_dim):
                self.assertAlmostEqual(self.s['next_obs'][i,0],
                                       self.s['next_obs'][i,j])

    def test_done(self):
        self._check_ndarray(self.s['done'],1,(self.N_sample,),"done")

class TestPyPrioritizedReplayBuffer(unittest.TestCase):
    """=== PyPrioritizedReplayBuffer.py ==="""

    obs_dim = 3
    act_dim = 1

    N_step = 100
    N_buffer_size = 32
    N_sample = 16

    alpha = 0.7
    beta = 0.5

    N_time = 1000

    @classmethod
    def setUpClass(cls):
        cls.rb = ReplayBuffer.PyPrioritizedReplayBuffer(cls.N_buffer_size,
                                                        cls.alpha,
                                                        cls.obs_dim,
                                                        cls.act_dim)
        for i in range(cls.N_step):
            cls.rb.add(np.ones(shape=(1,cls.obs_dim))*i,
                       np.zeros(shape=(1,cls.act_dim)),
                       0.5*i * np.ones((1)),
                       np.ones(shape=(1,cls.obs_dim))*(i+1),
                       np.zeros((1),dtype=np.int32) if i is not cls.N_step - 1 else np.ones((1),dtype=np.int32),
                       1)
        cls.s = cls.rb.sample(cls.N_sample,cls.beta)

        start = time.perf_counter()
        for _ in range(cls.N_time):
            cls.rb.sample(cls.N_sample,cls.beta)
        end = time.perf_counter()
        print("PER Sample {} time execution".format(cls.N_time))
        print("{} s".format(end - start))

    def _check_ndarray(self,array,ndim,shape,name):
        self.assertEqual(ndim,array.ndim)
        self.assertEqual(shape,array.shape)
        print("PER " + name + " {}".format(array))

    def test_obs(self):
        self._check_ndarray(self.s['obs'],min(2,self.obs_dim),
                            (self.N_sample,
                             self.obs_dim) if self.obs_dim > 1 else (self.N_sample,),
                            "obs")

    def test_act(self):
        self._check_ndarray(self.s['act'],min(2,self.act_dim),
                            (self.N_sample,
                             self.act_dim) if self.act_dim > 1 else (self.N_sample,),
                            "act")

    def test_rew(self):
        self._check_ndarray(self.s['rew'],1,(self.N_sample,),"rew")

    def test_next_obs(self):
        self._check_ndarray(self.s['next_obs'],min(2,self.obs_dim),
                            (self.N_sample,
                             self.obs_dim) if self.obs_dim > 1 else (self.N_sample,),
                            "next_obs")

        for i in range(self.N_sample):
            self.assertGreaterEqual(self.s['next_obs'][i,0],
                                    self.N_step - self.N_buffer_size)
            self.assertLess(self.s['next_obs'][i,0],self.N_step+1)

            for j in range(1,self.obs_dim):
                self.assertAlmostEqual(self.s['next_obs'][i,0],
                                       self.s['next_obs'][i,j])

    def test_done(self):
        self._check_ndarray(self.s['done'],1,(self.N_sample,),"done")

    def test_weights(self):
        self._check_ndarray(self.s['weights'],1,(self.N_sample,),"weights")

    def test_indexes(self):
        self._check_ndarray(self.s['indexes'],1,(self.N_sample,),"indexes")

if __name__ == '__main__':
    unittest.main()
