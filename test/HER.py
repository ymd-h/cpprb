import unittest

import numpy as np

from cpprb import HindsightReplayBuffer

class TestHER(unittest.TestCase):
    def test_get_buffer_size(self):
        buffer_size = 10
        hrb = HindsightReplayBuffer(size=buffer_size,
                                    env_dict={"obs": {}, "act": {}, "next_obs": {}},
                                    max_episode_len=2,
                                    reward_func=lambda s,a,g: -1*(s==g),
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
                                    reward_func=lambda s,a,g: -1*(s==g),
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


if __name__ == "__main__":
    unittest.main()
