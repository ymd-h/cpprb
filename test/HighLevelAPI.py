import numpy as np
import unittest

from cpprb import create_buffer,explore
from cpprb import (ReplayBuffer,PrioritizedReplayBuffer,
                   NstepReplayBuffer,NstepPrioritizedReplayBuffer,
                   ProcessSharedReplayBuffer,ProcessSharedPrioritizedReplayBuffer)

class TestCreateBuffer(unittest.TestCase):
    def test_class(self):
        size = 256
        obs_dim = 15
        act_dim = 3

        def cb_type(**kwargs):
            return type(create_buffer(size,obs_dim,act_dim,**kwargs))

        self.assertIs(cb_type(), ReplayBuffer)

        self.assertIs(cb_type(prioritized = True), PrioritizedReplayBuffer)

        self.assertIs(cb_type(Nstep = True), NstepReplayBuffer)

        self.assertIs(cb_type(prioritized = True, Nstep = True),
                      NstepPrioritizedReplayBuffer)

        self.assertIs(cb_type(process_shared = True), ProcessSharedReplayBuffer)

        self.assertIs(cb_type(process_shared = True, prioritized = True),
                      ProcessSharedPrioritizedReplayBuffer)

        with self.assertRaises(NotImplementedError):
            cb_type(process_shared = True, Nstep = True)

        with self.assertRaises(NotImplementedError):
            cb_type(process_shared = True, Nstep = True, prioritized = True)


class TestExplore(unittest.TestCase):
    def test_explore(self):
        size = 1024
        obs_dim = 7
        act_dim = 3
        rew_dim = 2

        rb1 = create_buffer(size,obs_dim,act_dim,rew_dim=rew_dim)
        rb2 = create_buffer(size,obs_dim,act_dim,rew_dim=rew_dim)

        def policy_stub(*args,**kwargs):
            return np.ones((act_dim),np.double)

        class env_stub:
            def reset(self):
                return np.ones((obs_dim),np.double)

            def step(self,*args,**kwargs):
                return (np.ones((obs_dim),np.double),
                        np.ones((rew_dim),np.double),
                        np.zeros(1,np.double),
                        None)

        env = env_stub()

        n_iteration = 256
        episode_len = 2

        explore(rb1,policy_stub,env,n_iteration,longest_step = episode_len)

        for it in range(n_iteration):
            o = env.reset()
            for _ in range(episode_len):
                a = policy_stub(o)
                no, r, d, _ = env.step(a)
                rb2.add(o,a,r,no,d)

                o = no

        idx = np.arange(size)
        np.testing.assert_allclose(rb1._encode_sample(idx)["obs"],
                                   rb2._encode_sample(idx)["obs"])

if __name__ == '__main__':
    unittest.main()
