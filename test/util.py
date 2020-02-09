import unittest

import gym

from cpprb import ReplayBuffer
from cpprb.util import create_env_dict

class TestAlgorithms(unittest.TestCase):
    def test_create_dict(self):
        env_dict = create_env_dict(self.env)

        self.assertIn("obs",env_dict)
        self.assertIn("act",env_dict)

    def test_add(self):
        env_dict = create_env_dict(self.env)
        rb = ReplayBuffer(256,env_dict)

        obs = self.env.reset()

        for i in range(100):
            act = self.env.action_space.sample()
            next_obs, rew, done, _ = self.env.step(act)

            rb.add(obs=obs,
                   act=act,
                   next_obs=obs,
                   rew=rew,
                   done=done)

            obs = next_obs


class TestCopy(TestAlgorithms):
    def setUp(self):
        self.env = gym.make("Copy-v0")

class TestDuplicatedInput(TestAlgorithms):
    def setUp(self):
        self.env = gym.make("DuplicatedInput-v0")

class TestRepeatCopy(TestAlgorithms):
    def setUp(self):
        self.env = gym.make("RepeatCopy-v0")

class TestReverse(TestAlgorithms):
    def setUp(self):
        self.env = gym.make("Reverse-v0")

class TestReversedAddition(TestAlgorithms):
    def setUp(self):
        self.env = gym.make("ReversedAddition-v0")

class TestReversedAddition3(TestAlgorithms):
    def setUp(self):
        self.env = gym.make("ReversedAddition3-v0")



if __name__ == "__main__":
    unittest.main()
