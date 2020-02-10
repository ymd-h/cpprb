import unittest

import gym

from cpprb import ReplayBuffer, create_env_dict, create_before_add_func
from cpprb.gym import NotebookAnimation


class TestEnv:
    @classmethod
    def setUpClass(cls):
        cls.display = NotebookAnimation()

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
        self.env = gym.make("Pendulum-v0")

class TestBipedalWalker(TestEnv,unittest.TestCase):
    def setUp(self):
        self.env = gym.make("BipedalWalker-v3")

class TestBipedalWalkerHardcore(TestEnv,unittest.TestCase):
    def setUp(self):
        self.env = gym.make("BipedalWalkerHardcore-v3")

class TestCarRacing(TestEnv,unittest.TestCase):
    def setUp(self):
        self.env = gym.make("CarRacing-v0")

class TestLunarLander(TestEnv,unittest.TestCase):
    def setUp(self):
        self.env = gym.make("LunarLander-v2")

class TestLunarLanderContinuous(TestEnv,unittest.TestCase):
    def setUp(self):
        self.env = gym.make("LunarLanderContinuous-v2")

class TestBlackjack(TestEnv,unittest.TestCase):
    def setUp(self):
        self.env = gym.make("Blackjack-v0")

class TestFrozenLake(TestEnv,unittest.TestCase):
    def setUp(self):
        self.env = gym.make("FrozenLake-v0")

class TestFrozenLake8x8(TestEnv,unittest.TestCase):
    def setUp(self):
        self.env = gym.make("FrozenLake8x8-v0")

class TestGuessingGame(TestEnv,unittest.TestCase):
    def setUp(self):
        self.env = gym.make("GuessingGame-v0")

class TestHotterColder(TestEnv,unittest.TestCase):
    def setUp(self):
        self.env = gym.make("HotterColder-v0")

class TestNChain(TestEnv,unittest.TestCase):
    def setUp(self):
        self.env = gym.make("NChain-v0")

class TestRoulette(TestEnv,unittest.TestCase):
    def setUp(self):
        self.env = gym.make("Roulette-v0")

class TestTaxi(TestEnv,unittest.TestCase):
    def setUp(self):
        self.env = gym.make("Taxi-v3")

class TestKellyCoinflip(TestEnv,unittest.TestCase):
    def setUp(self):
        self.env = gym.make("KellyCoinflip-v0")

class TestKellyCoinflipGeneralized(TestEnv,unittest.TestCase):
    def setUp(self):
        self.env = gym.make("KellyCoinflipGeneralized-v0")

class TestCliffWalking(TestEnv,unittest.TestCase):
    def setUp(self):
        self.env = gym.make("CliffWalking-v0")

# Robotics
class TestFetchSlide(TestEnv,unittest.TestCase):
    def setUp(self):
        self.env = gym.make("FetchSlide-v1")

class TestFetchSlideDense(TestEnv,unittest.TestCase):
    def setUp(self):
        self.env = gym.make("FetchSlideDense-v1")

class TestFetchPickAndPlace(TestEnv,unittest.TestCase):
    def setUp(self):
        self.env = gym.make("FetchPickAndPlace-v1")

class TestFetchPickAndPlaceDense(TestEnv,unittest.TestCase):
    def setUp(self):
        self.env = gym.make("FetchPickAndPlaceDense-v1")

class TestFetchReach(TestEnv,unittest.TestCase):
    def setUp(self):
        self.env = gym.make("FetchReach-v1")

class TestFetchReachDense(TestEnv,unittest.TestCase):
    def setUp(self):
        self.env = gym.make("FetchReachDense-v1")

class TestFetchPush(TestEnv,unittest.TestCase):
    def setUp(self):
        self.env = gym.make("FetchPush-v1")

class TestFetchPushDense(TestEnv,unittest.TestCase):
    def setUp(self):
        self.env = gym.make("FetchPushDense-v1")

class TestHandReach(TestEnv,unittest.TestCase):
    def setUp(self):
        self.env = gym.make("HandReach-v0")

class TestHandReachDense(TestEnv,unittest.TestCase):
    def setUp(self):
        self.env = gym.make("HandReachDense-v0")

class TestHandManipulateBlockRotateZ(TestEnv,unittest.TestCase):
    def setUp(self):
        self.env = gym.make("HandManipulateBlockRotateZ-v0")

class TestHandManipulateBlockRotateZDense(TestEnv,unittest.TestCase):
    def setUp(self):
        self.env = gym.make("HandManipulateBlockRotateZDense-v0")

class TestHandManipulateBlockRotateZTouchSensors(TestEnv,unittest.TestCase):
    def setUp(self):
        self.env = gym.make("HandManipulateBlockRotateZTouchSensors-v0")

class TestHandManipulateBlockRotateZTouchSensorsDense(TestEnv,unittest.TestCase):
    def setUp(self):
        self.env = gym.make("HandManipulateBlockRotateZTouchSensorsDense-v0")

class TestHandManipulateBlockRotateZTouchSensors1(TestEnv,unittest.TestCase):
    def setUp(self):
        self.env = gym.make("HandManipulateBlockRotateZTouchSensors-v1")

class TestHandManipulateBlockRotateZTouchSensorsDense1(TestEnv,unittest.TestCase):
    def setUp(self):
        self.env = gym.make("HandManipulateBlockRotateZTouchSensorsDense-v1")

class TestHandManipulateBlockRotateParallel(TestEnv,unittest.TestCase):
    def setUp(self):
        self.env = gym.make("HandManipulateBlockRotateParallel-v0")

class TestHandManipulateBlockRotateParallelDense(TestEnv,unittest.TestCase):
    def setUp(self):
        self.env = gym.make("HandManipulateBlockRotateParallelDense-v0")

class TestHandManipulateBlockRotateParallelTouchSensors(TestEnv,unittest.TestCase):
    def setUp(self):
        self.env = gym.make("HandManipulateBlockRotateParallelTouchSensors-v1")

class TestHandManipulateBlockRotateParallelTouchSensorsDense(TestEnv,unittest.TestCase):
    def setUp(self):
        self.env = gym.make("HandManipulateBlockRotateParallelTouchSensorsDense-v1")

class TestHandManipulateBlockRotateXYZ(TestEnv,unittest.TestCase):
    def setUp(self):
        self.env = gym.make("HandManipulateBlockRotateXYZ-v0")

class TestHandManipulateBlockRotateXYZDense(TestEnv,unittest.TestCase):
    def setUp(self):
        self.env = gym.make("HandManipulateBlockRotateXYZDense-v0")

class TestHandManipulateBlockRotateXYZTouchSensors(TestEnv,unittest.TestCase):
    def setUp(self):
        self.env = gym.make("HandManipulateBlockRotateXYZTouchSensors-v0")

class TestHandManipulateBlockRotateXYZTouchSensorsDense(TestEnv,unittest.TestCase):
    def setUp(self):
        self.env = gym.make("HandManipulateBlockRotateXYZTouchSensorsDense-v0")

class TestHandManipulateBlockRotateXYZTouchSensors1(TestEnv,unittest.TestCase):
    def setUp(self):
        self.env = gym.make("HandManipulateBlockRotateXYZTouchSensors-v1")

class TestHandManipulateBlockRotateXYZTouchSensorsDense1(TestEnv,unittest.TestCase):
    def setUp(self):
        self.env = gym.make("HandManipulateBlockRotateXYZTouchSensorsDense-v1")

class TestHandManipulateBlockFull(TestEnv,unittest.TestCase):
    def setUp(self):
        self.env = gym.make("HandManipulateBlockFull-v0")

class TestHandManipulateBlockFullDense(TestEnv,unittest.TestCase):
    def setUp(self):
        self.env = gym.make("HandManipulateBlockFullDense-v0")

class TestHandManipulateBlock(TestEnv,unittest.TestCase):
    def setUp(self):
        self.env = gym.make("HandManipulateBlock-v0")

class TestHandManipulateBlockDense(TestEnv,unittest.TestCase):
    def setUp(self):
        self.env = gym.make("HandManipulateBlockDense-v0")

class TestHandManipulateBlockTouchSensors(TestEnv,unittest.TestCase):
    def setUp(self):
        self.env = gym.make("HandManipulateBlockTouchSensors-v0")

class TestHandManipulateBlockTouchSensorsDense(TestEnv,unittest.TestCase):
    def setUp(self):
        self.env = gym.make("HandManipulateBlockTouchSensorsDense-v0")

class TestHandManipulateBlockTouchSensors1(TestEnv,unittest.TestCase):
    def setUp(self):
        self.env = gym.make("HandManipulateBlockTouchSensors-v1")

class TestHandManipulateBlockTouchSensorsDense1(TestEnv,unittest.TestCase):
    def setUp(self):
        self.env = gym.make("HandManipulateBlockTouchSensorsDense-v1")

class TestHandManipulateEggRotate(TestEnv,unittest.TestCase):
    def setUp(self):
        self.env = gym.make("HandManipulateEggRotate-v0")

class TestHandManipulateEggRotateDense(TestEnv,unittest.TestCase):
    def setUp(self):
        self.env = gym.make("HandManipulateEggRotateDense-v0")

class TestHandManipulateEggRotateTouchSensors(TestEnv,unittest.TestCase):
    def setUp(self):
        self.env = gym.make("HandManipulateEggRotateTouchSensors-v0")

class TestHandManipulateEggRotateTouchSensorsDense(TestEnv,unittest.TestCase):
    def setUp(self):
        self.env = gym.make("HandManipulateEggRotateTouchSensorsDense-v0")

class TestHandManipulateEggFull(TestEnv,unittest.TestCase):
    def setUp(self):
        self.env = gym.make("HandManipulateEggFull-v0")

class TestHandManipulateEggFullDense(TestEnv,unittest.TestCase):
    def setUp(self):
        self.env = gym.make("HandManipulateEggFullDense-v0")

class TestHandManipulateEgg(TestEnv,unittest.TestCase):
    def setUp(self):
        self.env = gym.make("HandManipulateEgg-v0")

class TestHandManipulateEggDense(TestEnv,unittest.TestCase):
    def setUp(self):
        self.env = gym.make("HandManipulateEggDense-v0")

class TestHandManipulateEggTouchSensors(TestEnv,unittest.TestCase):
    def setUp(self):
        self.env = gym.make("HandManipulateEggTouchSensors-v0")

class TestHandManipulateEggTouchSensorsDense(TestEnv,unittest.TestCase):
    def setUp(self):
        self.env = gym.make("HandManipulateEggTouchSensorsDense-v0")

class TestHandManipulateEggTouchSensors1(TestEnv,unittest.TestCase):
    def setUp(self):
        self.env = gym.make("HandManipulateEggTouchSensors-v1")

class TestHandManipulateEggTouchSensorsDense1(TestEnv,unittest.TestCase):
    def setUp(self):
        self.env = gym.make("HandManipulateEggTouchSensorsDense-v1")

class TestHandManipulatePenRotate(TestEnv,unittest.TestCase):
    def setUp(self):
        self.env = gym.make("HandManipulatePenRotate-v0")

class TestHandManipulatePenRotateDense(TestEnv,unittest.TestCase):
    def setUp(self):
        self.env = gym.make("HandManipulatePenRotateDense-v0")

class TestHandManipulatePenRotateTouchSensors(TestEnv,unittest.TestCase):
    def setUp(self):
        self.env = gym.make("HandManipulatePenRotateTouchSensors-v0")

class TestHandManipulatePenRotateTouchSensorsDense(TestEnv,unittest.TestCase):
    def setUp(self):
        self.env = gym.make("HandManipulatePenRotateTouchSensorsDense-v0")

class TestHandManipulatePenRotateTouchSensors1(TestEnv,unittest.TestCase):
    def setUp(self):
        self.env = gym.make("HandManipulatePenRotateTouchSensors-v1")

class TestHandManipulatePenRotateTouchSensorsDens1(TestEnv,unittest.TestCase):
    def setUp(self):
        self.env = gym.make("HandManipulatePenRotateTouchSensorsDense-v1")

class TestHandManipulatePenFull(TestEnv,unittest.TestCase):
    def setUp(self):
        self.env = gym.make("HandManipulatePenFull-v0")

class TestHandManipulatePenFullDense(TestEnv,unittest.TestCase):
    def setUp(self):
        self.env = gym.make("HandManipulatePenFullDense-v0")

class TestHandManipulatePen(TestEnv,unittest.TestCase):
    def setUp(self):
        self.env = gym.make("HandManipulatePen-v0")

class TestHandManipulatePenDense(TestEnv,unittest.TestCase):
    def setUp(self):
        self.env = gym.make("HandManipulatePenDense-v0")

class TestHandManipulatePenTouchSensors(TestEnv,unittest.TestCase):
    def setUp(self):
        self.env = gym.make("HandManipulatePenTouchSensors-v0")

class TestHandManipulatePenTouchSensorsDense(TestEnv,unittest.TestCase):
    def setUp(self):
        self.env = gym.make("HandManipulatePenTouchSensorsDense-v0")

class TestHandManipulatePenTouchSensors1(TestEnv,unittest.TestCase):
    def setUp(self):
        self.env = gym.make("HandManipulatePenTouchSensors-v1")

class TestHandManipulatePenTouchSensorsDense1(TestEnv,unittest.TestCase):
    def setUp(self):
        self.env = gym.make("HandManipulatePenTouchSensorsDense-v1")


if __name__ == "__main__":
    unittest.main()
