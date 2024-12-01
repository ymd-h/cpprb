import base64
import multiprocessing as mp
import unittest
import sys

from cpprb import MPReplayBuffer, MPPrioritizedReplayBuffer
import numpy as np


@unittest.skipUnless((3, 8) <= sys.version_info <= (3, 12),
                     "Support Ray only for Python 3.8-3.11")
class TestRay(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import ray

        cls.m = mp.get_context().Manager()
        ray.init()

    @classmethod
    def tearDownClass(cls):
        import ray

        ray.shutdown()
        cls.m.shutdown()

    def test_er(self):
        import ray

        rb = MPReplayBuffer(10, {"done": {}}, ctx=self.m, backend="SharedMemory")

        @ray.remote
        class RemoteWorker:
            authkey = base64.b64encode(mp.current_process().authkey)

            def __init__(self):
                mp.current_process().authkey = base64.b64decode(self.authkey)

            def add(self, rb):
                rb.add(done=0)
                rb.add(done=1)


            def sample(self, rb):
                return rb.sample(2)

        w = RemoteWorker.remote()

        ray.get([w.add.remote(rb)])
        self.assertEqual(rb.get_stored_size(), 2)
        np.testing.assert_equal(rb.get_all_transitions()["done"].ravel(),
                                np.asarray([0, 1]))


        s = ray.get(w.sample.remote(rb))
        self.assertIn("done", s)
        self.assertEqual(s["done"].shape[0], 2)

    def test_per(self):
        import ray

        rb = MPPrioritizedReplayBuffer(10, {"done": {}},
                                       ctx=self.m, backend="SharedMemory")


        @ray.remote
        class RemoteWorker:
            authkey = base64.b64encode(mp.current_process().authkey)

            def __init__(self):
                mp.current_process().authkey = base64.b64decode(self.authkey)

            def add(self, rb):
                rb.add(done=0)
                rb.add(done=1, priorities=0.1)

            def sample(self, rb):
                return rb.sample(2)

        w = RemoteWorker.remote()

        ray.get([w.add.remote(rb)])
        self.assertEqual(rb.get_stored_size(), 2)
        np.testing.assert_equal(rb.get_all_transitions()["done"].ravel(),
                                np.asarray([0, 1]))


        s = ray.get(w.sample.remote(rb))
        self.assertIn("done", s)
        self.assertIn("weights", s)
        self.assertIn("indexes", s)
        self.assertEqual(s["done"].shape[0], 2)

    def test_with_alpha(self):
        import ray

        rb = MPPrioritizedReplayBuffer(10, {"done": {}}, alpha=0.2,
                                       ctx=self.m, backend="SharedMemory")

        @ray.remote
        class RemoteWorker:
            authkey = base64.b64encode(mp.current_process().authkey)

            def __init__(self):
                mp.current_process().authkey = base64.b64decode(self.authkey)

            def add(self, rb):
                rb.add(done=[0, 1])

            def get_all_transitions(self, rb):
                return rb.get_all_transitions()

        w = RemoteWorker.remote()

        ray.get([w.add.remote(rb), w.add.remote(rb)])

        np.testing.assert_equal(rb.get_all_transitions()["done"],
                                ray.get(w.get_all_transitions.remote(rb))["done"])
        self.assertEqual(rb.get_stored_size(), 4)
        np.testing.assert_equal(rb.get_all_transitions()["done"].ravel(),
                                np.asarray([0, 1, 0, 1]))


if __name__ == "__main__":
    unittest.main()
