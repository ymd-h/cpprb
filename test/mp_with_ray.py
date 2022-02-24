import base64
import multiprocessing as mp
import unittest
import sys

from cpprb import MPReplayBuffer, MPPrioritizedReplayBuffer
import numpy as np
import ray


@unittest.skipUnless(sys.version_info >= (3,8), "Support Ray only for Python 3.8+")
class TestRay(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m = mp.get_context().Manager()
        ray.init()

        authkey = base64.b64encode(mp.current_process().authkey)

        def auth_fn(*args):
            mp.current_process().authkey = base64.b64decode(authkey)

        ray.worker.global_worker.run_function_on_all_workers(auth_fn)

    @classmethod
    def tearDownClass(cls):
        ray.shutdown()
        cls.m.shutdown()

    def test_er(self):
        rb = MPReplayBuffer(10, {"done": {}}, ctx=self.m, backend="SharedMemory")

        @ray.remote
        def add(rb):
            rb.add(done=0)
            rb.add(done=1)

        ray.get([add.remote(rb)])
        self.assertEqual(rb.get_stored_size(), 2)
        np.testing.assert_equal(rb.get_all_transitions()["done"].ravel(),
                                np.asarray([0, 1]))

        @ray.remote
        def sample(rb):
            return rb.sample(2)

        s = ray.get(sample.remote(rb))
        self.assertIn("done", s)
        self.assertEqual(s["done"].shape[0], 2)

    def test_per(self):
        rb = MPPrioritizedReplayBuffer(10, {"done": {}},
                                       ctx=self.m, backend="SharedMemory")

        @ray.remote
        def add(rb):
            rb.add(done=0)
            rb.add(done=1, priorities=0.1)

        ray.get([add.remote(rb)])
        self.assertEqual(rb.get_stored_size(), 2)
        np.testing.assert_equal(rb.get_all_transitions()["done"].ravel(),
                                np.asarray([0, 1]))

        @ray.remote
        def sample(rb):
            return rb.sample(2)

        s = ray.get(sample.remote(rb))
        self.assertIn("done", s)
        self.assertIn("weights", s)
        self.assertIn("indexes", s)
        self.assertEqual(s["done"].shape[0], 2)

    def test_with_alpha(self):
        rb = MPPrioritizedReplayBuffer(10, {"done": {}}, alpha=0.2,
                                       ctx=self.m, backend="SharedMemory")

        @ray.remote
        def add(rb):
            rb.add(done=[0, 1])

        @ray.remote
        def get_all_transitions(rb):
            return rb.get_all_transitions()

        ray.get([add.remote(rb), add.remote(rb)])

        np.testing.assert_equal(rb.get_all_transitions()["done"],
                                ray.get(get_all_transitions.remote(rb))["done"])
        self.assertEqual(rb.stored_size(), 4)
        np.testing.assert_equal(rb.get_all_transitions()["done"].ravel(),
                                np.asarray([0, 1, 0, 1]))


if __name__ == "__main__":
    unittest.main()
