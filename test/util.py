import os
import platform
import sys
import unittest

import gym

has_algorithmic = sys.version_info < (3,10)
has_legacy_toytext = sys.version_info < (3,10)

if has_algorithmic:
    import gym_algorithmic

if has_legacy_toytext:
    import gym_toytext

if platform.system() == 'Linux':
    import pyvirtualdisplay

from cpprb import ReplayBuffer, create_env_dict, create_before_add_func

if platform.system() == 'Linux':
    display = pyvirtualdisplay.Display()
    display.start()

@unittest.skipIf(os.getenv("GITHUB_ACTIONS"),"Skip on GitHub Actions")
@unittest.skipUnless(platform.system() == 'Linux',"Test only on Linux")
class TestEnv:
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

class TestAlgorithms(TestEnv):
    def test_create_dict(self):
        env_dict = create_env_dict(self.env)

        self.assertIn("obs",env_dict)
        self.assertIn("act0",env_dict)
        self.assertIn("act1",env_dict)
        self.assertIn("act2",env_dict)

@unittest.skipUnless(has_algorithmic,
                     "gym-algorithmic supports < Python 3.10")
class TestCopy(TestAlgorithms,unittest.TestCase):
    def setUp(self):
        self.env = gym.make("Copy-v0")

@unittest.skipUnless(has_algorithmic,
                     "gym-algorithmic supports < Python 3.10")
class TestDuplicatedInput(TestAlgorithms,unittest.TestCase):
    def setUp(self):
        self.env = gym.make("DuplicatedInput-v0")

@unittest.skipUnless(has_algorithmic,
                     "gym-algorithmic supports < Python 3.10")
class TestRepeatCopy(TestAlgorithms,unittest.TestCase):
    def setUp(self):
        self.env = gym.make("RepeatCopy-v0")

@unittest.skipUnless(has_algorithmic,
                     "gym-algorithmic supports < Python 3.10")
class TestReverse(TestAlgorithms,unittest.TestCase):
    def setUp(self):
        self.env = gym.make("Reverse-v0")

@unittest.skipUnless(has_algorithmic,
                     "gym-algorithmic supports < Python 3.10")
class TestReversedAddition(TestAlgorithms,unittest.TestCase):
    def setUp(self):
        self.env = gym.make("ReversedAddition-v0")

@unittest.skipUnless(has_algorithmic,
                     "gym-algorithmic supports < Python 3.10")
class TestReversedAddition3(TestAlgorithms,unittest.TestCase):
    def setUp(self):
        self.env = gym.make("ReversedAddition3-v0")

class TestAcrobot(TestEnv,unittest.TestCase):
    def setUp(self):
        self.env = gym.make("Acrobot-v1")

class TestCartPole(TestEnv,unittest.TestCase):
    def setUp(self):
        self.env = gym.make("CartPole-v1")

class TestMountainCar(TestEnv,unittest.TestCase):
    def setUp(self):
        self.env = gym.make("MountainCar-v0")

class TestMountainCarContinuous(TestEnv,unittest.TestCase):
    def setUp(self):
        self.env = gym.make("MountainCarContinuous-v0")

class TestPendulum(TestEnv,unittest.TestCase):
    def setUp(self):
        self.env = gym.make("Pendulum-v1")

class TestBipedalWalker(TestEnv,unittest.TestCase):
    def setUp(self):
        self.env = gym.make("BipedalWalker-v3")

class TestBipedalWalkerHardcore(TestEnv,unittest.TestCase):
    def setUp(self):
        self.env = gym.make("BipedalWalkerHardcore-v3")

class TestCarRacing(TestEnv,unittest.TestCase):
    def setUp(self):
        self.env = gym.make("CarRacing-v1")

class TestLunarLander(TestEnv,unittest.TestCase):
    def setUp(self):
        self.env = gym.make("LunarLander-v2")

class TestLunarLanderContinuous(TestEnv,unittest.TestCase):
    def setUp(self):
        self.env = gym.make("LunarLanderContinuous-v2")

class TestBlackjack(TestEnv,unittest.TestCase):
    def setUp(self):
        self.env = gym.make("Blackjack-v1")

class TestFrozenLake(TestEnv,unittest.TestCase):
    def setUp(self):
        self.env = gym.make("FrozenLake-v1")

class TestFrozenLake8x8(TestEnv,unittest.TestCase):
    def setUp(self):
        self.env = gym.make("FrozenLake8x8-v1")

@unittest.skipUnless(has_legacy_toytext,
                     "gym-legacy-toytext supports < Python 3.10")
class TestGuessingGame(TestEnv,unittest.TestCase):
    def setUp(self):
        self.env = gym.make("GuessingGame-v0")

class TestHotterColder(TestEnv,unittest.TestCase):
    def setUp(self):
        self.env = gym.make("HotterColder-v0")

@unittest.skipUnless(has_legacy_toytext,
                     "gym-legacy-toytext supports < Python 3.10")
class TestNChain(TestEnv,unittest.TestCase):
    def setUp(self):
        self.env = gym.make("NChain-v0")

@unittest.skipUnless(has_legacy_toytext,
                     "gym-legacy-toytext supports < Python 3.10")
class TestRoulette(TestEnv,unittest.TestCase):
    def setUp(self):
        self.env = gym.make("Roulette-v0")

class TestTaxi(TestEnv,unittest.TestCase):
    def setUp(self):
        self.env = gym.make("Taxi-v3")

@unittest.skipUnless(has_legacy_toytext,
                     "gym-legacy-toytext supports < Python 3.10")
class TestKellyCoinflip(TestEnv,unittest.TestCase):
    def setUp(self):
        self.env = gym.make("KellyCoinflip-v0")

@unittest.skipUnless(has_legacy_toytext,
                     "gym-legacy-toytext supports < Python 3.10")
class TestKellyCoinflipGeneralized(TestEnv,unittest.TestCase):
    def setUp(self):
        self.env = gym.make("KellyCoinflipGeneralized-v0")

class TestCliffWalking(TestEnv,unittest.TestCase):
    def setUp(self):
        self.env = gym.make("CliffWalking-v0")



if __name__ == "__main__":
    unittest.main()
