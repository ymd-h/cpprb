import unittest

import numpy as np

from cpprb import (MPReplayBuffer as ReplayBuffer,
                   MPPrioritizedReplayBuffer as PrioritizedReplayBuffer)

class TestReplayBuffer(unittest.TestCase):
    def test_buffer(self):

        buffer_size = 256
        obs_shape = (15,15)
        act_dim = 5

        N = 512

        erb = ReplayBuffer(buffer_size,{"obs":{"shape": obs_shape},
                                        "act":{"shape": act_dim},
                                        "rew":{},
                                        "next_obs":{"shape": obs_shape},
                                        "done":{}})

        for i in range(N):
            obs = np.full(obs_shape,i,dtype=np.double)
            act = np.full(act_dim,i,dtype=np.double)
            rew = i
            next_obs = obs + 1
            done = 0

            erb.add(obs=obs,act=act,rew=rew,next_obs=next_obs,done=done)

        es = erb._encode_sample(range(buffer_size))

        erb.sample(32)

        erb.clear()

        self.assertEqual(erb.get_next_index(),0)
        self.assertEqual(erb.get_stored_size(),0)

    def test_add(self):
        buffer_size = 256
        obs_shape = (15,15)
        act_dim = 5

        rb = ReplayBuffer(buffer_size,{"obs":{"shape": obs_shape},
                                       "act":{"shape": act_dim},
                                       "rew":{},
                                       "next_obs": {"shape": obs_shape},
                                       "done": {}})

        self.assertEqual(rb.get_next_index(),0)
        self.assertEqual(rb.get_stored_size(),0)

        obs = np.zeros(obs_shape)
        act = np.ones(act_dim)
        rew = 1
        next_obs = obs + 1
        done = 0

        rb.add(obs=obs,act=act,rew=rew,next_obs=next_obs,done=done)

        self.assertEqual(rb.get_next_index(),1)
        self.assertEqual(rb.get_stored_size(),1)

        with self.assertRaises(KeyError):
            rb.add(obs=obs)

        self.assertEqual(rb.get_next_index(),1)
        self.assertEqual(rb.get_stored_size(),1)

        obs = np.stack((obs,obs))
        act = np.stack((act,act))
        rew = (1,0)
        next_obs = np.stack((next_obs,next_obs))
        done = (0.0,1.0)

        rb.add(obs=obs,act=act,rew=rew,next_obs=next_obs,done=done)

        self.assertEqual(rb.get_next_index(),3)
        self.assertEqual(rb.get_stored_size(),3)

    def test_default_dtype(self):
        buffer_size = 256

        rb = ReplayBuffer(buffer_size,{"done": {}},
                          default_dtype = np.float32)

        rb.add(done=1)
        self.assertEqual(rb.sample(1)["done"][0].dtype,np.float32)

class TestPrioritizedReplayBuffer(unittest.TestCase):
    def test_add(self):
        buffer_size = 500
        obs_shape = (84,84,3)
        act_dim = 10

        rb = PrioritizedReplayBuffer(buffer_size,{"obs": {"shape": obs_shape},
                                                  "act": {"shape": act_dim},
                                                  "rew": {},
                                                  "done": {}})

        obs = np.zeros(obs_shape)
        act = np.ones(act_dim)
        rew = 1
        done = 0

        rb.add(obs=obs,act=act,rew=rew,done=done)

        ps = 1.5

        rb.add(obs=obs,act=act,rew=rew,done=done,priorities=ps)

        self.assertAlmostEqual(rb.get_max_priority(),1.5)

        obs = np.stack((obs,obs))
        act = np.stack((act,act))
        rew = (1,0)
        done = (0.0,1.0)

        rb.add(obs=obs,act=act,rew=rew,done=done)

        ps = (0.2,0.4)
        rb.add(obs=obs,act=act,rew=rew,done=done,priorities=ps)


        rb.clear()
        self.assertEqual(rb.get_next_index(),0)
        self.assertEqual(rb.get_stored_size(),0)

    def test_sample(self):
        buffer_size = 500
        obs_shape = (84,84,3)
        act_dim = 4

        rb = PrioritizedReplayBuffer(buffer_size,{"obs": {"shape": obs_shape},
                                                  "act": {"shape": act_dim},
                                                  "rew": {},
                                                  "done": {}})

        obs = np.zeros(obs_shape)
        act = np.ones(act_dim)
        rew = 1
        done = 0

        rb.add(obs=obs,act=act,rew=rew,done=done)

        ps = 1.5

        rb.add(obs=obs,act=act,rew=rew,done=done,priorities=ps)

        self.assertAlmostEqual(rb.get_max_priority(),1.5)

        obs = np.stack((obs,obs))
        act = np.stack((act,act))
        rew = (1,0)
        done = (0.0,1.0)

        rb.add(obs=obs,act=act,rew=rew,done=done)

        ps = (0.2,0.4)
        rb.add(obs=obs,act=act,rew=rew,done=done,priorities=ps)

        sample = rb.sample(64)

        w = sample["weights"]
        i = sample["indexes"]

        rb.update_priorities(i,w*w)

if __name__ == '__main__':
    unittest.main()
