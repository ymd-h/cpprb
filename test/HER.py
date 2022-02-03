import unittest

import numpy as np

from cpprb import HindsightReplayBuffer

class TestHER(unittest.TestCase):
    def test_get_buffer_size(self):
        buffer_size = 10
        hrb = HindsightReplayBuffer(size=buffer_size,
                                    env_dict={"obs": {}, "act": {}, "next_obs": {}},
                                    max_episode_len=2,
                                    reward_func=lambda s,a,g: -1*(s!=g),
                                    additonal_goals=1,
                                    prioritized=False)
        self.assertEqual(hrb.get_buffer_size(), buffer_size)

    def test_stored_size(self):
        """
        Test get_stored_size() method
        """
        hrb = HindsightReplayBuffer(size=10,
                                    env_dict={"obs": {}, "act": {}, "next_obs": {}},
                                    max_episode_len=2,
                                    reward_func=lambda s,a,g: -1*(s!=g),
                                    additonal_goals=1,
                                    prioritized=False)
        # Buffer is initialized without data
        self.assertEqual(hrb.get_stored_size(), 0)

        # During episode, stored size doesn't increase
        hrb.add(obs=0, act=0, next_obs=0)
        self.assertEqual(hrb.get_stored_size(), 0)

        # On episode end, stored size increases by `episode_len * additional_goals`
        hrb.on_episode_end(1)
        self.assertEqual(hrb.get_stored_size(), 2)

        # If no transactions in the current episode, nothing happens
        hrb.on_episode_end(1)
        self.assertEqual(hrb.get_stored_size(), 2)

        # By calling clear(), stored size become 0 again.
        hrb.clear()
        self.assertEqual(hrb.get_stored_size(), 0)

    def test_future(self):
        rew_func = lambda s,a,g: -1*(s!=g)
        hrb = HindsightReplayBuffer(size=10,
                                    env_dict={"obs": {}, "act": {}, "next_obs": {}},
                                    max_episode_len=2,
                                    strategy = "future",
                                    reward_func=rew_func,
                                    additonal_goals=2,
                                    prioritized=False)
        hrb.add(obs=0, act=0, next_obs=1)
        hrb.add(obs=1, act=0, next_obs=2)
        hrb.on_episode_end(3)
        self.assertEqual(hrb.get_stored_size(), 6)

        sample = hrb.get_all_transitions()
        self.assertIsIn("rew", sample)
        self.assertIsIn("goal", sample)
        self.assertEqual(sample["obs"].shape, (6,1))
        self.assertTrue((sample["obs"] < sample["goal"]).all())
        np.testing.assert_allclose(sample["rew"],
                                   rew_func(sample["obs"],
                                            sample["act"],
                                            sample["goal"]))

    def test_episode(self):
        rew_func = lambda s,a,g: -1*(s!=g)
        hrb = HindsightReplayBuffer(size=10,
                                    env_dict={"obs": {}, "act": {}, "next_obs": {}},
                                    max_episode_len=2,
                                    strategy = "episode",
                                    reward_func=rew_func,
                                    additonal_goals=2,
                                    prioritized=False)
        hrb.add(obs=0, act=0, next_obs=1)
        hrb.add(obs=1, act=0, next_obs=2)
        hrb.on_episode_end(3)
        self.assertEqual(hrb.get_stored_size(), 6)

        sample = hrb.get_all_transitions()
        self.assertIsIn("rew", sample)
        self.assertIsIn("goal", sample)
        self.assertEqual(sample["obs"].shape, (6,1))
        np.testing.assert_allclose(sample["rew"],
                                   rew_func(sample["obs"],
                                            sample["act"],
                                            sample["goal"]))


    def test_random(self):
        rew_func = lambda s,a,g: -1*(s!=g)
        hrb = HindsightReplayBuffer(size=10,
                                    env_dict={"obs": {}, "act": {}, "next_obs": {}},
                                    max_episode_len=2,
                                    strategy = "random",
                                    reward_func=rew_func,
                                    additonal_goals=2,
                                    prioritized=False)
        hrb.add(obs=0, act=0, next_obs=1)
        hrb.add(obs=1, act=0, next_obs=2)
        hrb.on_episode_end(3)
        self.assertEqual(hrb.get_stored_size(), 6)

        sample = hrb.get_all_transitions()
        self.assertIsIn("rew", sample)
        self.assertIsIn("goal", sample)
        self.assertEqual(sample["obs"].shape, (6,1))
        np.testing.assert_allclose(sample["rew"],
                                   rew_func(sample["obs"],
                                            sample["act"],
                                            sample["goal"]))

    def test_final(self):
        rew_func = lambda s,a,g: -1*(s!=g)
        hrb = HindsightReplayBuffer(size=10,
                                    env_dict={"obs": {}, "act": {}, "next_obs": {}},
                                    max_episode_len=2,
                                    strategy = "final",
                                    reward_func=rew_func,
                                    additonal_goals=2,
                                    prioritized=False)

        # additonal_goals is ignored
        self.assertEqual(hrb.additional_goals, 1)

        hrb.add(obs=0, act=0, next_obs=1)
        hrb.add(obs=1, act=0, next_obs=2)
        hrb.on_episode_end(3)
        self.assertEqual(hrb.get_stored_size(), 4)

        sample = hrb.get_all_transitions()
        self.assertIsIn("rew", sample)
        self.assertIsIn("goal", sample)
        self.assertEqual(sample["obs"].shape, (4,1))
        np.testing.assert_allclose(sample["goal"][2:],
                                   np.broadcast_to(2, (2,1)))


    def test_unknown_strategy(self):
        rew_func = lambda s,a,g: -1*(s!=g)
        with self.assertRaises(ValueError):
            hrb = HindsightReplayBuffer(size=10,
                                        env_dict={"obs": {},"act": {},"next_obs": {}},
                                        max_episode_len=2,
                                        strategy = "__UNKNOWN_STRATEGY__",
                                        reward_func=rew_func,
                                        additonal_goals=2,
                                        prioritized=False)


    def test_assert_exceed_max_episode(self):
        rew_func = lambda s,a,g: -1*(s!=g)
        hrb = HindsightReplayBuffer(size=10,
                                    env_dict={"obs": {}, "act": {}, "next_obs": {}},
                                    max_episode_len=2,
                                    strategy = "future",
                                    reward_func=rew_func,
                                    additonal_goals=2,
                                    prioritized=False)

        hrb.add(obs=0, act=0, next_obs=1)
        hrb.add(obs=1, act=0, next_obs=2)
        with self.assertRaises(ValueError):
            hrb.add(obs=2, act=0, next_obs=3)

    def test_assert_PER(self):
        rew_func = lambda s,a,g: -1*(s!=g)
        hrb = HindsightReplayBuffer(size=10,
                                    env_dict={"obs": {}, "act": {}, "next_obs": {}},
                                    max_episode_len=2,
                                    strategy = "future",
                                    reward_func=rew_func,
                                    additonal_goals=2,
                                    prioritized=False)

        hrb.add(obs=0, act=0, next_obs=1)
        hrb.add(obs=1, act=0, next_obs=2)

        with self.assertRaises(ValueError):
            hrb.get_max_priority()

        with self.assertRaises(ValueError):
            hrb.update_priorities([], [])

if __name__ == "__main__":
    unittest.main()
