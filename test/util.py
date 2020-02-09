import unittest

import gym

from cpprb import ReplayBuffer, create_env_dict, create_before_add_func

class TestAlgorithms:
    def test_create_dict(self):
        env_dict = create_env_dict(self.env)

        self.assertIn("obs",env_dict)
        self.assertIn("act0",env_dict)
        self.assertIn("act1",env_dict)
        self.assertIn("act2",env_dict)

    def test_add(self):
        env_dict = create_env_dict(self.env)
        before_add_func = create_before_add_func(self.env)

        rb = ReplayBuffer(256,env_dict)

        obs = self.env.reset()

        for i in range(100):
            act = self.env.action_space.sample()
            next_obs, rew, done, _ = self.env.step(act)

            rb.add(**before_add_func(obs,act,next_obs,rew,done))

            if done:
                obs = self.env.reset()
            else:
                obs = next_obs


class TestCopy(TestAlgorithms,unittest.TestCase):
    def setUp(self):
        self.env = gym.make("Copy-v0")

class TestDuplicatedInput(TestAlgorithms,unittest.TestCase):
    def setUp(self):
        self.env = gym.make("DuplicatedInput-v0")

class TestRepeatCopy(TestAlgorithms,unittest.TestCase):
    def setUp(self):
        self.env = gym.make("RepeatCopy-v0")

class TestReverse(TestAlgorithms,unittest.TestCase):
    def setUp(self):
        self.env = gym.make("Reverse-v0")

class TestReversedAddition(TestAlgorithms,unittest.TestCase):
    def setUp(self):
        self.env = gym.make("ReversedAddition-v0")

class TestReversedAddition3(TestAlgorithms,unittest.TestCase):
    def setUp(self):
        self.env = gym.make("ReversedAddition3-v0")



if __name__ == "__main__":
    unittest.main()
