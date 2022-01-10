import sys
import unittest

from cpprb import ReplayBuffer

@unittest.skipIf(sys.platform.startswith("win"), "JAX doesn't support Windows")
class TestJAX(unittest.TestCase):
    def test_add(self):
        import jax.numpy as jnp
        rb = ReplayBuffer(4, {"done": {}})

        done = jnp.asarray(1)

        for i in range(5):
            with self.subTest(i=i):
                rb.add(done=done)

    def test_multistep_add(self):
        import jax.numpy as jnp
        rb = ReplayBuffer(4, {"done": {}})

        done = jnp.asarray([1,1,1])

        for i in range(2):
            with self.subTest(i=i):
                rb.add(done=done)


    def test_nstep(self):
        import jax.numpy as jnp
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
        import jax.numpy as jnp
        rb = ReplayBuffer(6, {"obs": {}, "rew": {}, "done": {}, "next_obs":{}},
                          Nstep={"size": 4, "rew": "rew", "next": "next_obs"})

        obs = jnp.asarray([1,1,1,1])
        rew = jnp.asarray([1,1,1,1])
        done = jnp.asarray([1,1,1,1])
        next_obs = jnp.asarray([1,1,1,1])

        for i in range(7):
            with self.subTest(i=i):
                rb.add(obs=obs, rew=rew, done=done, next_obs=next_obs)


if __name__ == "__main__":
    unittest.main()
