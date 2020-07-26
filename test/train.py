import unittest

import numpy as np

from cpprb import ReplayBuffer, PrioritizedReplayBuffer, train

class Env:
    def __init__(self,shape=(1,)):
        self.shape = shape

    def reset(self):
        return np.zeros(self.shape)

    def step(self,action):
        return np.zeros(self.shape), 1.0, np.random.choice([0.0,1.0],p=[0.99,0.01]), {}


class TestTrain(unittest.TestCase):
    def setUp(self):
        self.env = Env(shape=(3,))


    def test_default_train(self):
        """
        Run train function with default arguments
        """
        rb = ReplayBuffer(32,
                          {"obs": {"shape": (3,)},
                           "act": {},
                           "rew": {},
                           "next_obs": {"shape": (3,)},
                           "done": {}})
        train(rb,self.env,
              lambda obs: 1.0,
              lambda kwargs,step,episode: 0.5,
              max_steps=10)

    def test_per_train(self):
        """
        Run train function with PER
        """
        rb = PrioritizedReplayBuffer(32,
                                     {"obs": {"shape": (3,)},
                                      "act": {},
                                      "rew": {},
                                      "next_obs": {"shape": (3,)},
                                      "done": {}})
        train(rb,self.env,
              lambda obs: 1.0,
              lambda kwargs,step,episode: 0.5,
              max_steps=10)

    def test_too_big_max_steps(self):
        """
        Raise ValueError for too big max_steps
        """
        rb = ReplayBuffer(32,
                          {"obs": {"shape": (3,)},
                           "act": {},
                           "rew": {},
                           "next_obs": {"shape": (3,)},
                           "done": {}})
        def update(kw,step,episode):
            raise RuntimeError

        with self.assertRaises(ValueError):
            train(rb,self.env,
                  lambda obs: 1.0,
                  update,
                  max_steps=int(1e+32))

    def test_update_count(self):
        """
        Check step and episode

        step < max_steps
        episode <= step
        """
        rb = ReplayBuffer(32,
                          {"obs": {"shape": (3,)},
                           "act": {},
                           "rew": {},
                           "next_obs": {"shape": (3,)},
                           "done": {}})
        def update(kw,step,episode):
            self.assertLess(step,10)
            self.assertLessEqual(episode,step)
            return 0.5

        train(rb,self.env,
              lambda obs: 1.0,
              update,
              max_steps=10)

    def test_warmup(self):
        """
        Skip warmup steps

        n_warmups <= step
        """
        rb = ReplayBuffer(32,
                          {"obs": {"shape": (3,)},
                           "act": {},
                           "rew": {},
                           "next_obs": {"shape": (3,)},
                           "done": {}})
        def update(kw,step,episode):
            self.assertGreaterEqual(step,5)
            self.assertLess(step,10)
            self.assertLessEqual(episode,step)
            return 0.5

        train(rb,self.env,
              lambda obs: 1.0,
              update,
              max_steps=10,
              n_warmups=5)

    def test_done_check(self):
        """
        Pass custom check_done which always return `True`

        Always step == episode
        """
        rb = ReplayBuffer(32,
                          {"obs": {"shape": (3,)},
                           "act": {},
                           "rew": {},
                           "next_obs": {"shape": (3,)},
                           "done": {}})
        def update(kw,step,episode):
            self.assertLess(step,10)
            self.assertEqual(step,episode)
            return 0.5

        train(rb,self.env,
              lambda obs: 1.0,
              update,
              max_steps=10,
              done_check=lambda kw: True)

    def test_per_without_TD(self):
        """
        Run train function with PER withou TD

        Raise TypeError
        """
        rb = PrioritizedReplayBuffer(32,
                                     {"obs": {"shape": (3,)},
                                      "act": {},
                                      "rew": {},
                                      "next_obs": {"shape": (3,)},
                                      "done": {}})
        with self.assertRaises(TypeError):
            train(rb,self.env,
                  lambda obs: 1.0,
                  lambda kwargs,step,episode: None,
                  max_steps=10)

    def test_after_step(self):
        """
        Pass custom after_step
        """
        rb = ReplayBuffer(32,
                          {"obs": {"shape": (3,)},
                           "act": {},
                           "rew": {},
                           "next_obs": {"shape": (3,)},
                           "done": {}})

        def after_step(obs,act,step_returns,step,episode):
            next_obs, rew, done, info = step_returns
            self.assertEqual(obs.shape,next_obs.shape)
            return {"obs": obs,
                    "act": act,
                    "next_obs": next_obs,
                    "rew": rew,
                    "done": done}

        def update(kw,step,episode):
            self.assertLess(step,10)
            return 0.5

        train(rb,self.env,
              lambda obs: 1.0,
              update,
              max_steps=10,
              after_step=after_step)

if __name__ == "__main__":
    unittest.main()
