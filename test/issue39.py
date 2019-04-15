import numpy as np
import unittest

from cpprb import ReplayBuffer


class TestIssue39(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.rb = ReplayBuffer(obs_dim=3, act_dim=3, size=10)
        for i in range(10):
            obs_act = np.array([i for _ in range(3)], dtype=np.float64)
            cls.rb.add(obs=obs_act,
                       act=obs_act,
                       next_obs=obs_act,
                       rew=float(i),
                       done=False)
        cls.s = cls.rb._encode_sample(range(10))

    def test_obs(self):
        self.assertTrue((self.s['obs'] == np.array(((0,0,0),
                                                    (1,1,1),
                                                    (2,2,2),
                                                    (3,3,3),
                                                    (4,4,4),
                                                    (5,5,5),
                                                    (6,6,6),
                                                    (7,7,7),
                                                    (8,8,8),
                                                    (9,9,9)))).all())
    
    def test_act(self):
        self.assertTrue((self.s['act'] == np.array(((0,0,0),
                                                    (1,1,1),
                                                    (2,2,2),
                                                    (3,3,3),
                                                    (4,4,4),
                                                    (5,5,5),
                                                    (6,6,6),
                                                    (7,7,7),
                                                    (8,8,8),
                                                    (9,9,9)))).all())
    def test_next_obs(self):
        self.assertTrue((self.s['next_obs'] == np.array(((0,0,0),
                                                         (1,1,1),
                                                         (2,2,2),
                                                         (3,3,3),
                                                         (4,4,4),
                                                         (5,5,5),
                                                         (6,6,6),
                                                         (7,7,7),
                                                         (8,8,8),
                                                         (9,9,9)))).all())
    def test_rew(self):
        self.assertTrue((self.s['rew'] == np.array((0,1,2,3,4,
                                                    5,6,7,8,9)).reshape(-1,1)).all())

    def test_done(self):
        self.assertTrue((self.s['done'] == np.zeros(shape=(10))).all())

if __name__ == '__main__':
    unittest.main()

