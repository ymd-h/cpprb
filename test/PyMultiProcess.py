import numpy as np
import unittest, time
from multiprocessing import Process
from cpprb import *

def timer(f,N_times,name,*args,**kwargs):
        start = time.perf_counter()
        for _ in range(N_times):
            f(*args,**kwargs)
        end = time.perf_counter()
        print("{}: {} time execution".format(name,N_times))
        print("{} s".format(end - start))


class TestMultiProcessReplayBuffer(unittest.TestCase):
    class_name = "MultiProcessing"
    buffer_size = 1024 * 256
    N_add = buffer_size * 3
    N_time = 10
    add_dim = 100

    @classmethod
    def setUpClass(cls):
        cls.rb = ThreadSafeReplayBuffer(cls.buffer_size, cls.obs_dim, cls.act_dim)

        def f(rb,end):
            obs = np.ones(shape=(cls.obs_dim))
            act = np.zeros(shape=(cls.act_dim))
            next_obs = np.ones(shape=(cls.obs_dim))
            for i in range(0,end):
                rb.add(obs,act,1.0,next_obs, 0)

        def g(rb,end):
            obs = np.ones(shape=(cls.add_dim,cls.obs_dim))
            act = np.zeros(shape=(cls.add_dim,cls.act_dim))
            rew = np.ones((cls.add_dim))
            next_obs = np.ones(shape=(cls.add_dim,cls.obs_dim))
            done = np.zeros((cls.add_dim))
            for i in range(0,end,cls.add_dim):
                rb.add(obs,act,rew,next_obs,done)

        def Multi_(_f):
            def func(rb,end):
                q = [Process(target=_f,
                             args=(rb.init_worker(),end //8,)) for _ in range(8)]
                for qe in q:
                    qe.start()
                for qe in q:
                    qe.join()
            return func

        def Multi_f(rb,end):
            Multi_(f)(rb,end)

        def Multi_g(rb,end):
            Multi_(g)(rb,end)

        print("Test for single thread with 1 point")
        timer(lambda: f(cls.rb,cls.N_add),cls.N_time,
              "Single thread adding 1 time-point")

        print("Test for single thread with 100 point")
        timer(lambda: g(cls.rb,cls.N_add),cls.N_time,
              "Single thread adding {} time-point".format(cls.add_dim))

        print("Test for multi thread with 1 point")
        timer(lambda: Multi_(f)(cls.rb,cls.N_add),cls.N_time,
              "Multi thread adding 1 time-point")

        print("Test for multi thread with 100 point")
        timer(lambda: Multi_(g)(cls.rb,cls.N_add),cls.N_time,
              "Multi thread adding {} time-point".format(cls.add_dim))

        cls.fill_ReplayBuffer()
        cls.s = cls.rb.sample(cls.batch_size)
