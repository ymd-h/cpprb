import sys
import unittest

import numpy as np

from cpprb import ReplayBuffer
from cpprb.PyReplayBuffer import NstepBuffer


is_win = sys.platform.startswith("win")
if not is_win:
    import jax.numpy as jnp


@unittest.skipIf(is_win, "JAX doesn't support Windows")
class TestJAX(unittest.TestCase):
    def test_nstep_buffer(self):
        buffer = NstepBuffer({"obs": {}, "rew": {},  "done": {}, "next_obs": {}},
                             Nstep={"size": 3, "rew": "rew", "next": "next_obs"})
        obs = jnp.asarray([1])
        rew = jnp.asarray([1])
        done = jnp.asarray([1])
        next_obs = jnp.asarray([1])

        for i in range(4):
            with self.subTest(i=i):
                buffer.add(obs=obs, rew=rew, done=done, next_obs=next_obs)

    def test_add(self):
        rb = ReplayBuffer(4, {"done": {}})

        done = jnp.asarray(1)

        for i in range(5):
            with self.subTest(i=i):
                rb.add(done=done)

    def test_multistep_add(self):
        rb = ReplayBuffer(4, {"done": {}})

        done = jnp.asarray([1,1,1])

        for i in range(2):
            with self.subTest(i=i):
                rb.add(done=done)


    def test_nstep(self):
        rb = ReplayBuffer(6, {"obs": {}, "rew": {}, "done": {}, "next_obs":{}},
                          Nstep={"size": 4, "rew": "rew", "next": "next_obs"})

        obs = jnp.asarray(1)
        rew = jnp.asarray(1)
        done = jnp.asarray(1)
        next_obs = jnp.asarray(1)

        for i in range(7):
            with self.subTest(i=i):
                rb.add(obs=obs, rew=rew, done=done, next_obs=next_obs)

    def tet_nstep_multistep_add(self):
        rb = ReplayBuffer(6, {"obs": {}, "rew": {}, "done": {}, "next_obs":{}},
                          Nstep={"size": 4, "rew": "rew", "next": "next_obs"})

        obs = jnp.asarray([1,1,1,1])
        rew = jnp.asarray([1,1,1,1])
        done = jnp.asarray([1,1,1,1])
        next_obs = jnp.asarray([1,1,1,1])

        for i in range(7):
            with self.subTest(i=i):
                rb.add(obs=obs, rew=rew, done=done, next_obs=next_obs)


class TestUnwriteable(unittest.TestCase):
    def test_nstep(self):
        rb = ReplayBuffer(6, {"obs": {}, "rew": {}, "done": {}, "next_obs":{}},
                          Nstep={"size": 4, "rew": "rew", "next": "next_obs"})

        obs = np.asarray(1)
        rew = np.asarray(1)
        done = np.asarray(1)
        next_obs = np.asarray(1)

        obs.flags.writeable = False
        rew.flags.writeable = False
        done.flags.writeable = False
        next_obs.flags.writeable = False

        for i in range(7):
            with self.subTest(i=i):
                rb.add(obs=obs, rew=rew, done=done, next_obs=next_obs)

    def tet_nstep_multistep_add(self):
        rb = ReplayBuffer(6, {"obs": {}, "rew": {}, "done": {}, "next_obs":{}},
                          Nstep={"size": 4, "rew": "rew", "next": "next_obs"})

        obs = np.asarray([1,1,1,1])
        rew = np.asarray([1,1,1,1])
        done = np.asarray([1,1,1,1])
        next_obs = np.asarray([1,1,1,1])

        obs.flags.writeable = False
        rew.flags.writeable = False
        done.flags.writeable = False
        next_obs.flags.writeable = False

        for i in range(7):
            with self.subTest(i=i):
                rb.add(obs=obs, rew=rew, done=done, next_obs=next_obs)


if __name__ == "__main__":
    unittest.main()
