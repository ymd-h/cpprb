import unittest

import gym

from cpprb import ReplayBuffer
from cpprb.util import create_env_dict

class TestAlgorithms(unittest.TestCase):
    @unittest.skipIf(type(self) == TestAlgorithms,
                     "TestAlgorithms is abstract class")
    def test_create_dict(self):
        env_dict = create_env_dict(self.env)

        self.assertIn("obs",env_dict)
        self.assertIn("act0",env_dict)
        self.assertIn("act1",env_dict)
        self.assertIn("act2",env_dict)

    @unittest.skipIf(type(self) == TestAlgorithms,
                     "TestAlgorithms is abstract class")
    def test_add(self):
        env_dict = create_env_dict(self.env)
        rb = ReplayBuffer(256,env_dict)

        obs = self.env.reset()

        for i in range(100):
            act = self.env.action_space.sample()
            next_obs, rew, done, _ = self.env.step(act)

            rb.add(obs=obs,
                   act0=act[0],
                   act1=act[1],
                   act2=act[2],
                   next_obs=obs,
                   rew=rew,
                   done=done)

            if done:
                obs = self.env.reset()
            else:
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
