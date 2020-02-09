import unittest

import gym

from cpprb import ReplayBuffer
from cpprb.util import create_env_dict

class TestAlgorithms(unittest.TestCase):
    def test_create_dict(self):
        env_dict = create_env_dict(self.env)

        self.assertIn("obs",env_dict)
        self.assertIn("act",env_dict)

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
