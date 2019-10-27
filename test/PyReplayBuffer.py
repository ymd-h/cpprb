import numpy as np
import unittest, time
from cpprb import *

class TestReplayBuffer(unittest.TestCase):
    """=== ReplayBuffer.py ==="""
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
            cls.rb.add(obs=np.ones(shape=(cls.obs_dim)),
                       act=np.zeros(shape=(cls.act_dim)),
                       rew=0.5,
                       next_obs=np.ones(shape=(cls.obs_dim)),
                       done=0)
        else:
            cls.rb.add(obs=np.ones(shape=(cls.obs_dim)),
                       act=np.zeros(shape=(cls.act_dim)),
                       rew=0.5,
                       next_obs=np.ones(shape=(cls.obs_dim)),
                       done=1)


        cls.rb.clear()

        for i in range(cls.N_add):
            cls.rb.add(obs=np.ones(shape=(cls.add_dim,cls.obs_dim))*i,
                       act=np.zeros(shape=(cls.add_dim,cls.act_dim)),
                       rew=np.ones((cls.add_dim)) * 0.5*i,
                       next_obs=np.ones(shape=(cls.add_dim,cls.obs_dim))*(i+1),
                       done=np.zeros((cls.add_dim)))
        else:
            cls.rb.add(obs=np.ones(shape=(cls.obs_dim)),
                       act=np.zeros(shape=(cls.act_dim)),
                       rew=0.5,
                       next_obs=np.ones(shape=(cls.obs_dim)),
                       done=1)

    @classmethod
    def setUpClass(cls):
        cls.rb = ReplayBuffer(cls.buffer_size,
                              {"obs": {"shape": cls.obs_dim},
                               "act": {"shape": cls.act_dim},
                               "rew": {},
                               "next_obs":{"shape": cls.obs_dim},
                               "done": {}})

        cls.fill_ReplayBuffer()
        cls.s = cls.rb.sample(cls.batch_size)

    def _check_ndarray(self,array,ndim,shape,name):
        self.assertEqual(ndim,array.ndim)
        self.assertEqual(shape,array.shape)
        print(self.class_name + ": " + name + " {}".format(array))

    def test_1_obs(self):
        self._check_ndarray(self.s['obs'],2,
                            (self.batch_size, self.obs_dim),
                            "obs")

    def test_2_act(self):
        self._check_ndarray(self.s['act'],2,
                            (self.batch_size, self.act_dim),
                            "act")

    def test_3_rew(self):
        self._check_ndarray(self.s['rew'],2,(self.batch_size,1),"rew")

    def test_4_next_obs(self):
        self._check_ndarray(self.s['next_obs'],2,
                            (self.batch_size, self.obs_dim),
                            "next_obs")

    def test_5_done(self):
        self._check_ndarray(self.s['done'],2,(self.batch_size,1),"done")
        for d in self.s['done']:
            self.assertIn(d,[0,1])

class TestPrioritizedBase:
    def test_weights(self):
        self._check_ndarray(self.s['weights'],1,(self.batch_size,),"weights")
        for w in self.s['weights']:
            self.assertAlmostEqual(w,1.0)

    def test_indexes(self):
        self._check_ndarray(self.s['indexes'],1,(self.batch_size,),"indexes")

    def test_priority_add(self):
        for i in range(self.N_add):
            self.rb2.add(obs=np.ones(shape=(self.obs_dim))*i,
                         act=np.zeros(shape=(self.act_dim)),
                         rew=0.5*i,
                         next_obs=np.ones(shape=(self.obs_dim))*(i+1),
                         done=0.0,
                         priorities=0.0 + 1e-6)
        for i in range(self.N_add):
            self.rb2.add(obs=np.ones(shape=(self.add_dim,self.obs_dim))*i,
                         act=np.zeros(shape=(self.add_dim,self.act_dim)),
                         rew=np.ones((self.add_dim)) * 0.5*i,
                         next_obs=np.ones(shape=(self.add_dim,self.obs_dim))*(i+1),
                         done=np.zeros(shape=(self.add_dim)),
                         priorities=np.zeros(shape=(self.add_dim)))
        else:
            self.rb2.add(obs=np.ones(shape=(self.obs_dim)),
                         act=np.zeros(shape=(self.act_dim)),
                         rew=0.5,
                         next_obs=np.ones(shape=(self.obs_dim)),
                         done=1,
                         priorities=0.1)

        self.s2 = self.rb2.sample(self.batch_size)
        self._check_ndarray(self.s2['indexes'],1,(self.batch_size,),
                            "indexes [0,...,0.1]")
        self._check_ndarray(self.s2['weights'],1,(self.batch_size,),
                            "weights [0,...,0.1]")

        self.rb2.update_priorities(np.ones(shape=(1),dtype=np.uintp),
                                   np.ones(shape=(1))*0.5)
        self.s3 = self.rb2.sample(self.batch_size)
        self._check_ndarray(self.s3['indexes'],1,(self.batch_size,),
                            "indexes [0.5,...,0.1]")
        self._check_ndarray(self.s3['weights'],1,(self.batch_size,),
                            "weights [0.5,...,0.1]")

    def test_update_indexes(self):
        for i in range(self.N_add):
            self.rb_ui.add(obs=np.ones(shape=(self.add_dim,self.obs_dim))*i,
                           act=np.zeros(shape=(self.add_dim,self.act_dim)),
                           rew=np.ones((self.add_dim)) * 0.5*i,
                           next_obs=np.ones(shape=(self.add_dim,self.obs_dim))*(i+1),
                           done=np.zeros(shape=(self.add_dim)),
                           priorities=np.zeros(shape=(self.add_dim)))
        else:
            self.rb_ui.add(obs=np.ones(shape=(self.obs_dim)),
                           act=np.zeros(shape=(self.act_dim)),
                           rew=0.5,
                           next_obs=np.ones(shape=(self.obs_dim)),
                           done=1,
                           priorities=0.1)

        for i,type in enumerate([np.byte,np.ubyte,
                                 np.short,np.ushort,
                                 np.intc,np.uintc,
                                 np.int_,np.uint,
                                 np.longlong,np.ulonglong,
                                 np.half,np.single,
                                 np.double,np.longdouble,
                                 np.int8,np.int16,
                                 np.int32,np.int64,
                                 np.uint8,np.uint16,
                                 np.uint32,np.uint64,
                                 np.intp,np.uintp,
                                 np.float32,np.float64]):
            with self.subTest(dtype=type):
                Nmax = 128
                self.rb_ui.update_priorities(np.arange(0,Nmax,
                                                       dtype=type),
                                             np.ones(shape=(Nmax))*0.5)


class TestPrioritizedReplayBuffer(TestReplayBuffer,TestPrioritizedBase):
    """=== PrioritizedReplayBuffer.py ==="""
    class_name = "PER"

    @classmethod
    def setUpClass(cls):
        cls.rb = PrioritizedReplayBuffer(cls.buffer_size,
                                         {"obs": {"shape": cls.obs_dim},
                                          "act": {"shape": cls.act_dim},
                                          "rew": {},
                                          "next_obs": {"shape": cls.obs_dim},
                                          "done": {}},
                                         alpha=cls.alpha)
        cls.rb2 = PrioritizedReplayBuffer(cls.buffer_size,
                                          {"obs": {"shape": cls.obs_dim},
                                           "act": {"shape": cls.act_dim},
                                           "rew": {},
                                           "next_obs": {"shape": cls.obs_dim},
                                           "done": {}},
                                          alpha=cls.alpha)
        cls.rb_ui = PrioritizedReplayBuffer(cls.buffer_size,
                                            {"obs": {"shape": cls.obs_dim},
                                             "act": {"shape": cls.act_dim},
                                             "rew": {},
                                             "next_obs": {"shape": cls.obs_dim},
                                             "done": {}},
                                            alpha=cls.alpha)
        cls.fill_ReplayBuffer()
        cls.s = cls.rb.sample(cls.batch_size,cls.beta)

class TestNstepBase:
    def test_discounts(self):
        self._check_ndarray(self.s['discounts'],2,(self.batch_size,1),"discounts")
        for g,d in zip(self.s['discounts'],self.s['done']):
            if(d > 0.0):
                self.assertAlmostEqual(g,1.0)

class TestNstepReplayBuffer(TestReplayBuffer,TestNstepBase):
    """=== NstepReplayBuffer.py ==="""
    class_name = "N-ER"

    @classmethod
    def setUpClass(cls):
        cls.rb = ReplayBuffer(cls.buffer_size,
                              {"obs": {"shape": cls.obs_dim},
                               "act": {"shape": cls.act_dim},
                               "rew": {},
                               "next_obs": {"shape": cls.obs_dim},
                               "done": {}},
                              Nstep={"size": cls.nstep,
                                     "rew": "rew",
                                     "gamma": cls.discount})
        cls.fill_ReplayBuffer()
        cls.s = cls.rb.sample(cls.batch_size)

class TestNstepPrioritizedReplayBuffer(TestReplayBuffer,
                                       TestPrioritizedBase,TestNstepBase):
    """=== NstepPrioritizedReplayBuffer.py ==="""
    class_name = "N-PER"

    @classmethod
    def setUpClass(cls):
        cls.rb = NstepPrioritizedReplayBuffer(cls.buffer_size,
                                              {"obs": {"shape": cls.obs_dim},
                                               "act": {"shape": cls.act_dim},
                                               "rew": {},
                                               "next_obs": {"shape": cls.obs_dim},
                                               "done": {}},
                                              alpha=cls.alpha)
        cls.rb2 = NstepPrioritizedReplayBuffer(cls.buffer_size,
                                               {"obs": {"shape": cls.obs_dim},
                                                "act": {"shape": cls.act_dim},
                                                "rew": {},
                                                "next_obs": {"shape": cls.obs_dim},
                                                "done": {}},
                                               alpha=cls.alpha)
        cls.rb_ui = NstepPrioritizedReplayBuffer(cls.buffer_size,
                                                 {"obs": {"shape": cls.obs_dim},
                                                  "act": {"shape": cls.act_dim},
                                                  "rew": {},
                                                  "next_obs": {"shape": cls.obs_dim},
                                                  "done": {}},
                                                 alpha=cls.alpha)
        cls.fill_ReplayBuffer()
        cls.s = cls.rb.sample(cls.batch_size,cls.beta)

        start = time.perf_counter()
        for _ in range(cls.N_time):
            cls.rb.sample(cls.batch_size,cls.beta)
        end = time.perf_counter()
        print("N-PER Sample {} time execution".format(cls.N_time))
        print("{} s".format(end - start))

class TestSelectiveReplayBuffer(TestReplayBuffer):
    """=== SelectiveReplayBuffer ==="""
    class_name = "S-ER"

    @classmethod
    def setUpClass(cls):
        cls.rb = SelectiveReplayBuffer(cls.buffer_size,
                                       {"obs": {"shape": cls.obs_dim},
                                        "act": {"shape": cls.act_dim},
                                        "rew": {},
                                        "next_obs": {"shape": cls.obs_dim},
                                        "done": {}},
                                       Nepisodes=10)
        cls.fill_ReplayBuffer()
        cls.s = cls.rb.sample(cls.batch_size)

    def test_episode(self):
        self.srb = SelectiveReplayBuffer(self.buffer_size,
                                         self.obs_dim,
                                         self.act_dim,
                                         Nepisodes=10)

        for i in range(self.N_add):
            self.srb.add(np.ones(shape=(self.add_dim,self.obs_dim))*i,
                         np.zeros(shape=(self.add_dim,self.act_dim)),
                         np.ones((self.add_dim)) * 0.5*i,
                         np.ones(shape=(self.add_dim,self.obs_dim))*(i+1),
                         np.random.randint(0,2,size=self.add_dim)*1.0)

        self.assertEqual(self.srb.get_next_index(),
                         min(self.N_add*self.add_dim,self.srb.get_buffer_size()))

        old_index = self.srb.get_next_index()
        s = self.srb.get_episode(2)
        delete_len = self.srb.delete_episode(2)
        self.assertEqual(self.srb.get_next_index(), old_index - delete_len)

if __name__ == '__main__':
    unittest.main()
