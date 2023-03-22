import os
import platform
import sys
import unittest

from cpprb import ReplayBuffer

try:
    from cpprb.util import (
        create_env_dict,
        create_before_add_func,
        Env,
        Discrete,
        Box,
        Tuple,
    )
except ImportError:
    has_gym = False
else:
    has_gym = True
    try:
        import gymnasium as gym
    except ImportError:
        import gym


skipUnlessGym = unittest.skipUnless(has_gym, "gym(nasium) is required")

def make(name):
    for i in range(5):
        try:
            return gym.make(name)
        except gym.error.DeprecatedEnv as e:
            base, v = name.split("-")
            name = f"{base}-v{(int(v[1:]) + 1)}"
    raise e


class TestEnv:
    def test_add(self):
        env_dict = create_env_dict(self.env)
        before_add_func = create_before_add_func(self.env)

        rb = ReplayBuffer(10, env_dict)

        obs = self.env.observation_space.sample()
        act = self.env.action_space.sample()

        rb.add(**before_add_func(obs, act, obs, 0.5, False))


@skipUnlessGym
class TestAlgorithmic(TestEnv, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        class AlgorithmicEnv(Env):
            def __init__(self):
                self.action_space = Tuple(
                    [Discrete(3), Discrete(2), Discrete(3)]
                )
                self.observation_space = Discrete(4)
        cls.env = AlgorithmicEnv()

@skipUnlessGym
class TestAcrobot(TestEnv, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env = make("Acrobot-v1")

@skipUnlessGym
class TestCartPole(TestEnv,unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env = make("CartPole-v1")

@skipUnlessGym
class TestMountainCar(TestEnv,unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env = make("MountainCar-v0")

@skipUnlessGym
class TestMountainCarContinuous(TestEnv,unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env = make("MountainCarContinuous-v0")

@skipUnlessGym
class TestPendulum(TestEnv,unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env = make("Pendulum-v1")

@skipUnlessGym
class TestBipedalWalker(TestEnv,unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env = make("BipedalWalker-v3")

@skipUnlessGym
class TestBipedalWalkerHardcore(TestEnv,unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env = make("BipedalWalkerHardcore-v3")

@skipUnlessGym
class TestCarRacing(TestEnv,unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env = make("CarRacing-v1")

@skipUnlessGym
class TestLunarLander(TestEnv,unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env = make("LunarLander-v2")

@skipUnlessGym
class TestLunarLanderContinuous(TestEnv,unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env = make("LunarLanderContinuous-v2")

@skipUnlessGym
class TestBlackjack(TestEnv,unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env = make("Blackjack-v1")

@skipUnlessGym
class TestFrozenLake(TestEnv,unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env = make("FrozenLake-v1")

@skipUnlessGym
class TestFrozenLake8x8(TestEnv,unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env = make("FrozenLake8x8-v1")


if __name__ == "__main__":
    unittest.main()
