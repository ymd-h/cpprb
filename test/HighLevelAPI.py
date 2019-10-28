import numpy as np
import unittest

from cpprb import create_buffer
from cpprb import (ReplayBuffer,PrioritizedReplayBuffer)

class TestCreateBuffer(unittest.TestCase):
    def test_class(self):
        size = 256
        obs_dim = 15
        act_dim = 3

        def cb_type(**kwargs):
            return type(create_buffer(size,
                                      {"obs": {"shape": obs_dim},
                                       "act": {"shape": act_dim},
                                       "rew": {},
                                       "next_obs": {"shape": obs_dim},
                                       "done": {}},**kwargs))

        self.assertIs(cb_type(), ReplayBuffer)

        self.assertIs(cb_type(prioritized = True), PrioritizedReplayBuffer)

if __name__ == '__main__':
    unittest.main()
