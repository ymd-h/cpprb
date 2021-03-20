import unittest

import numpy as np

from cpprb import ReplayBuffer, PrioritizedReplayBuffer


class TestReplayBuffer(unittest.TestCase):
    def test_basic(self):
        """
        Basic Test Case

        Loaded buffer have same transitions with saved one.
        """
        buffer_size = 4
        env_dict = {"a": {}}

        rb1 = ReplayBuffer(buffer_size, env_dict)
        rb2 = ReplayBuffer(buffer_size, env_dict)

        a = [1, 2, 3, 4]

        rb1.add(a=a)

        fname = "basic.npz"
        rb1.save_transitions(fname)
        rb2.load_transitions(fname)

        t1 = rb1.get_all_transitions()
        t2 = rb2.get_all_transitions()

        np.testing.assert_allclose(t1["a"], t2["a"])

    def test_smaller_buffer(self):
        """
        Load to smaller buffer

        Loaded buffer only stored last buffer_size transitions
        """
        buffer_size1 = 10
        buffer_size2 = 4
        env_dict = {"a": {}}

        rb1 = ReplayBuffer(buffer_size1, env_dict)
        rb2 = ReplayBuffer(buffer_size2, env_dict)

        a = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

        fname = "smaller.npz"
        rb1.save_transitions(fname)
        rb2.load_transitions(fname)

        t1 = rb1.get_all_transitions()
        t2 = rb2.get_all_transitions()

        np.testing.assert_allclose(t1["a"][-buffer_size2:],t2["a"])

    def test_load_to_filled_buffer(self):
        """
        Load to already filled buffer

        Add to transitions
        """
        buffer_size1 = 10
        buffer_size2 = 10
        env_dict = {"a": {}}

        rb1 = ReplayBuffer(buffer_size1, env_dict)
        rb2 = ReplayBuffer(buffer_size2, env_dict)

        a = [1, 2, 3, 4]
        b = [5, 6]

        rb1.add(a=a)
        rb2.add(a=b)

        fname="filled.npz"
        rb1.save_transitions(fname)
        rb2.load_transitions(fname)

        t1 = rb1.get_all_transitions()
        t2 = rb2.get_all_transitions()

        np.testing.assert_allclose(t1["a"], t2["a"][len(b):])

    def test_load_Nstep(self):
        """
        Load Nstep transitions
        """
        buffer_size = 10
        env_dict = {"done": {}}
        Nstep = {"size": 3, "gamma": 0.99}

        rb1 = ReplayBuffer(buffer_size, env_dict, Nstep=Nstep)
        rb2 = ReplayBuffer(buffer_size, env_dict, Nstep=Nstep)

        d = [0, 0, 0, 0, 1]

        rb1.add(done=d)
        rb1.on_episode_end()

        fname="Nstep.npz"
        rb1.save_transitions(fname)
        rb2.load_transitions(fname)

        t1 = rb1.get_all_transitions()
        t2 = rb2.get_all_transitions()

        np.testing.assert_allclose(t1["done"], t2["done"])

    def test_Nstep_incompatibility(self):
        """
        Raise ValueError when Nstep incompatibility
        """
        buffer_size = 10
        env_dict = {"done": {}}
        Nstep = {"size": 3, "gamma": 0.99}

        rb1 = ReplayBuffer(buffer_size, env_dict, Nstep=Nstep)
        rb2 = ReplayBuffer(buffer_size, env_dict)

        d = [0, 0, 0, 0, 1]

        rb1.add(done=d)
        rb1.on_episode_end()

        fname="Nstep_raise.npz"
        rb1.save_transitions(fname)

        with self.assertRaises(ValueError):
            rb2.load_transitions(fname)

    def test_next_of(self):
        """
        Load next_of transitions with safe mode

        For safe mode, next_of is not neccessary at loaded buffer.
        """
        buffer_size = 10
        env_dict = {"a": {}}

        rb1 = ReplayBuffer(buffer_size, env_dict, next_of="a")
        rb2 = ReplayBuffer(buffer_size, env_dict)

        a = [1, 2, 3, 4, 5, 6]

        rb1.add(a=a[:-1], next_a=a[1:])

        fname="next_of.npz"
        rb1.save_transitions(fname)
        rb2.load_transitions(fname)

        t1 = rb1.get_all_transitions()
        t2 = rb2.get_all_transitions()

        np.testing.assert_allclose(t1["a"], t2["a"])

    def test_unsafe_next_of(self):
        """
        Load next_of transitions with unsafe mode
        """
        buffer_size = 10
        env_dict = {"a": {}}

        rb1 = ReplayBuffer(buffer_size, env_dict, next_of="a")
        rb2 = ReplayBuffer(buffer_size, env_dict, next_of="a")

        a = [1, 2, 3, 4, 5, 6]

        rb1.add(a=a[:-1], next_a=a[1:])

        fname="unsafe_next_of.npz"
        rb1.save_transitions(fname, safe=False)
        rb2.load_transitions(fname)

        t1 = rb1.get_all_transitions()
        t2 = rb2.get_all_transitions()

        np.testing.assert_allclose(t1["a"], t2["a"])

    def test_raise_unsafe_next_of(self):
        """
        Load incompatible next_of transitions with unsafe mode
        """
        buffer_size = 10
        env_dict = {"a": {}}

        rb1 = ReplayBuffer(buffer_size, env_dict, next_of="a")
        rb2 = ReplayBuffer(buffer_size, env_dict)

        a = [1, 2, 3, 4, 5, 6]

        rb1.add(a=a[:-1], next_a=a[1:])

        fname="unsafe_incompatible_next_of.npz"
        rb1.save_transitions(fname, safe=False)

        with self.assertRaises(ValueError):
            rb2.load_transitions(fname)

if __name__ == "__main__":
    unittest.main()
