import numpy as np
import unittest, time
from cpprb import ReplayBuffer

class TestPyReplayBuffer(unittest.TestCase):
    """=== PyReplayBuffer.py ==="""
    class_name = "ER"

    obs_dim = 3
    act_dim = 1

    buffer_size = 1024
    add_dim = 10
    N_add = round(3.27 * buffer_size)
    batch_size = 16

    nstep = 4
    discount = 0.99

    alpha = 0.7
    beta = 0.5

    N_time = 1000

    @classmethod
    def fill_ReplayBuffer(cls):
        for i in range(cls.N_add):
            cls.rb.add(np.ones(shape=(cls.obs_dim)),
                       np.zeros(shape=(cls.act_dim)),
                       0.5,
                       np.ones(shape=(cls.obs_dim)),
                       0)
        else:
            cls.rb.add(np.ones(shape=(cls.obs_dim)),
                       np.zeros(shape=(cls.act_dim)),
                       0.5,
                       np.ones(shape=(cls.obs_dim)),
                       1)


        cls.rb.clear()

        for i in range(cls.N_add):
            cls.rb.add(np.ones(shape=(cls.add_dim,cls.obs_dim))*i,
                       np.zeros(shape=(cls.add_dim,cls.act_dim)),
                       np.ones((cls.add_dim)) * 0.5*i,
                       np.ones(shape=(cls.add_dim,cls.obs_dim))*(i+1),
                       np.zeros((cls.add_dim)))
        else:
            cls.rb.add(np.ones(shape=(cls.obs_dim)),
                       np.zeros(shape=(cls.act_dim)),
                       0.5,
                       np.ones(shape=(cls.obs_dim)),
                       1)

    @classmethod
    def setUpClass(cls):
        cls.rb = ReplayBuffer.PyReplayBuffer(cls.buffer_size,
                                             cls.obs_dim,
                                             cls.act_dim)

        cls.fill_ReplayBuffer()
        cls.s = cls.rb.sample(cls.batch_size)

    def _check_ndarray(self,array,ndim,shape,name):
        self.assertEqual(ndim,array.ndim)
        self.assertEqual(shape,array.shape)
        print(self.class_name + ": " + name + " {}".format(array))

    def test_obs(self):
        self._check_ndarray(self.s['obs'],2,
                            (self.batch_size, self.obs_dim),
                            "obs")

    def test_act(self):
        self._check_ndarray(self.s['act'],2,
                            (self.batch_size, self.act_dim),
                            "act")

    def test_rew(self):
        self._check_ndarray(self.s['rew'],1,(self.batch_size,),"rew")

    def test_next_obs(self):
        self._check_ndarray(self.s['next_obs'],2,
                            (self.batch_size, self.obs_dim),
                            "next_obs")

    def test_done(self):
        self._check_ndarray(self.s['done'],1,(self.batch_size,),"done")
        for d in self.s['done']:
            self.assertIn(d,[0,1])

class TestPrioritizedBase:
    def test_weights(self):
        self._check_ndarray(self.s['weights'],1,(self.batch_size,),"weights")
        for w in self.s['weights']:
            self.assertAlmostEqual(w,1.0)

    def test_indexes(self):
        self._check_ndarray(self.s['indexes'],1,(self.batch_size,),"indexes")


class TestPyPrioritizedReplayBuffer(TestPyReplayBuffer,TestPrioritizedBase):
    """=== PyPrioritizedReplayBuffer.py ==="""
    class_name = "PER"

    @classmethod
    def setUpClass(cls):
        cls.rb = ReplayBuffer.PyPrioritizedReplayBuffer(cls.buffer_size,
                                                        cls.obs_dim,
                                                        cls.act_dim,
                                                        alpha=cls.alpha)
        cls.fill_ReplayBuffer()
        cls.s = cls.rb.sample(cls.batch_size,cls.beta)

        start = time.perf_counter()
        for _ in range(cls.N_time):
            cls.rb.sample(cls.batch_size,cls.beta)
        end = time.perf_counter()
        print("PER Sample {} time execution".format(cls.N_time))
        print("{} s".format(end - start))


class TestNstepBase:
    def test_discounts(self):
        self._check_ndarray(self.s['discounts'],1,(self.batch_size,),"discounts")
        for g,d in zip(self.s['discounts'],self.s['done']):
            if(d > 0.0):
                self.assertAlmostEqual(g,1.0)

class TestPyNstepReplayBuffer(TestPyReplayBuffer,TestNstepBase):
    """=== PyNstepReplayBuffer.py ==="""
    class_name = "N-ER"

    @classmethod
    def setUpClass(cls):
        cls.rb = ReplayBuffer.PyNstepReplayBuffer(cls.buffer_size,
                                                  cls.obs_dim,
                                                  cls.act_dim,
                                                  nstep = cls.nstep,
                                                  discount = cls.discount)
        cls.fill_ReplayBuffer()
        cls.s = cls.rb.sample(cls.batch_size)

class TestPyNstepPrioritizedReplayBuffer(TestPyReplayBuffer,
                                         TestPrioritizedBase,TestNstepBase):
    """=== PyNstepPrioritizedReplayBuffer.py ==="""
    class_name = "N-PER"

    @classmethod
    def setUpClass(cls):
        cls.rb = ReplayBuffer.PyNstepPrioritizedReplayBuffer(cls.buffer_size,
                                                        cls.obs_dim,
                                                        cls.act_dim,
                                                        alpha=cls.alpha)
        cls.fill_ReplayBuffer()
        cls.s = cls.rb.sample(cls.batch_size,cls.beta)

        start = time.perf_counter()
        for _ in range(cls.N_time):
            cls.rb.sample(cls.batch_size,cls.beta)
        end = time.perf_counter()
        print("N-PER Sample {} time execution".format(cls.N_time))
        print("{} s".format(end - start))

if __name__ == '__main__':
    unittest.main()
