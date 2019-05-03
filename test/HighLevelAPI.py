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
