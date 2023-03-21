import os
import platform
import sys
import unittest

has_gym = sys.version_info < (3,11)
skipUnlessGym = unittest.skipUnless(has_gym, "gym supports < 3.11")

has_algorithmic = sys.version_info < (3,10)
has_legacy_toytext = sys.version_info < (3,10)

if has_gym:
    import gym
    from cpprb import create_env_dict, create_before_add_func

if has_algorithmic:
    # Register Copy, DuplicatedInput, RepeatCopy, Reverse, ReversedAddition
    import gym_algorithmic

if has_legacy_toytext:
    # Register GuessingGame, NChain, Roulette, KellyCoinflip, KellyCoinflipGeneralized
    import gym_toytext

if platform.system() == 'Linux':
    import pyvirtualdisplay

from cpprb import ReplayBuffer

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

@skipUnlessGym
@unittest.skipUnless(has_algorithmic,
                     "gym-algorithmic supports < Python 3.10")
class TestCopy(TestAlgorithms,unittest.TestCase):
    def setUp(self):
        self.env = gym.make("Copy-v0")

@skipUnlessGym
@unittest.skipUnless(has_algorithmic,
                     "gym-algorithmic supports < Python 3.10")
class TestDuplicatedInput(TestAlgorithms,unittest.TestCase):
    def setUp(self):
        self.env = gym.make("DuplicatedInput-v0")

@skipUnlessGym
@unittest.skipUnless(has_algorithmic,
                     "gym-algorithmic supports < Python 3.10")
class TestRepeatCopy(TestAlgorithms,unittest.TestCase):
    def setUp(self):
        self.env = gym.make("RepeatCopy-v0")

@skipUnlessGym
@unittest.skipUnless(has_algorithmic,
                     "gym-algorithmic supports < Python 3.10")
class TestReverse(TestAlgorithms,unittest.TestCase):
    def setUp(self):
        self.env = gym.make("Reverse-v0")

@skipUnlessGym
@unittest.skipUnless(has_algorithmic,
                     "gym-algorithmic supports < Python 3.10")
class TestReversedAddition(TestAlgorithms,unittest.TestCase):
    def setUp(self):
        self.env = gym.make("ReversedAddition-v0")

@skipUnlessGym
@unittest.skipUnless(has_algorithmic,
                     "gym-algorithmic supports < Python 3.10")
class TestReversedAddition3(TestAlgorithms,unittest.TestCase):
    def setUp(self):
        self.env = gym.make("ReversedAddition3-v0")

@skipUnlessGym
class TestAcrobot(TestEnv,unittest.TestCase):
    def setUp(self):
        self.env = gym.make("Acrobot-v1")

@skipUnlessGym
class TestCartPole(TestEnv,unittest.TestCase):
    def setUp(self):
        self.env = gym.make("CartPole-v1")

@skipUnlessGym
class TestMountainCar(TestEnv,unittest.TestCase):
    def setUp(self):
        self.env = gym.make("MountainCar-v0")

@skipUnlessGym
class TestMountainCarContinuous(TestEnv,unittest.TestCase):
    def setUp(self):
        self.env = gym.make("MountainCarContinuous-v0")

@skipUnlessGym
class TestPendulum(TestEnv,unittest.TestCase):
    def setUp(self):
        self.env = gym.make("Pendulum-v1")

@skipUnlessGym
class TestBipedalWalker(TestEnv,unittest.TestCase):
    def setUp(self):
        self.env = gym.make("BipedalWalker-v3")

@skipUnlessGym
class TestBipedalWalkerHardcore(TestEnv,unittest.TestCase):
    def setUp(self):
        self.env = gym.make("BipedalWalkerHardcore-v3")

@skipUnlessGym
class TestCarRacing(TestEnv,unittest.TestCase):
    def setUp(self):
        self.env = gym.make("CarRacing-v1")

@skipUnlessGym
class TestLunarLander(TestEnv,unittest.TestCase):
    def setUp(self):
        self.env = gym.make("LunarLander-v2")

@skipUnlessGym
class TestLunarLanderContinuous(TestEnv,unittest.TestCase):
    def setUp(self):
        self.env = gym.make("LunarLanderContinuous-v2")

@skipUnlessGym
class TestBlackjack(TestEnv,unittest.TestCase):
    def setUp(self):
        self.env = gym.make("Blackjack-v1")

@skipUnlessGym
class TestFrozenLake(TestEnv,unittest.TestCase):
    def setUp(self):
        self.env = gym.make("FrozenLake-v1")

@skipUnlessGym
class TestFrozenLake8x8(TestEnv,unittest.TestCase):
    def setUp(self):
        self.env = gym.make("FrozenLake8x8-v1")

@skipUnlessGym
@unittest.skipUnless(has_legacy_toytext,
                     "gym-legacy-toytext supports < Python 3.10")
class TestGuessingGame(TestEnv,unittest.TestCase):
    def setUp(self):
        self.env = gym.make("GuessingGame-v0")

@skipUnlessGym
class TestHotterColder(TestEnv,unittest.TestCase):
    def setUp(self):
        self.env = gym.make("HotterColder-v0")

@skipUnlessGym
@unittest.skipUnless(has_legacy_toytext,
                     "gym-legacy-toytext supports < Python 3.10")
class TestNChain(TestEnv,unittest.TestCase):
    def setUp(self):
        self.env = gym.make("NChain-v0")

@skipUnlessGym
@unittest.skipUnless(has_legacy_toytext,
                     "gym-legacy-toytext supports < Python 3.10")
class TestRoulette(TestEnv,unittest.TestCase):
    def setUp(self):
        self.env = gym.make("Roulette-v0")

@skipUnlessGym
class TestTaxi(TestEnv,unittest.TestCase):
    def setUp(self):
        self.env = gym.make("Taxi-v3")

@skipUnlessGym
@unittest.skipUnless(has_legacy_toytext,
                     "gym-legacy-toytext supports < Python 3.10")
class TestKellyCoinflip(TestEnv,unittest.TestCase):
    def setUp(self):
        self.env = gym.make("KellyCoinflip-v0")

@skipUnlessGym
@unittest.skipUnless(has_legacy_toytext,
                     "gym-legacy-toytext supports < Python 3.10")
class TestKellyCoinflipGeneralized(TestEnv,unittest.TestCase):
    def setUp(self):
        self.env = gym.make("KellyCoinflipGeneralized-v0")

@skipUnlessGym
class TestCliffWalking(TestEnv,unittest.TestCase):
    def setUp(self):
        self.env = gym.make("CliffWalking-v0")


if __name__ == "__main__":
    unittest.main()
