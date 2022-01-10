import sys
import unittest

from cpprb import ReplayBuffer

@unittest.skipIf(sys.platform.startswith("win"), "JAX doesn't support Windows")
class TestJAX(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import jax.numpy as jnp

    def test_add(self):
        rb = ReplayBuffer(32, {"done": {}})

        done = jnp.asarray(1)

        rb.add(done=done)

    def test_nstep(self):
        rb = ReplayBuffer(32, {"obs": {}, "rew": {}, "done": {}, "next_obs":{}},
                          Nstep={"size": 4, "rew": "rew", "next": "next_obs"})

        obs = jnp.asarray(1)
        rew = jnp.asarray(1)
        done = jnp.asarray(1)
        next_obs = jnp.asarray(1)

        rb.add(obs=obs, rew=rew, done=done, next_obs=next_obs)
        rb.add(obs=obs, rew=rew, done=done, next_obs=next_obs)
        rb.add(obs=obs, rew=rew, done=done, next_obs=next_obs)
        rb.add(obs=obs, rew=rew, done=done, next_obs=next_obs)


if __name__ == "__main__":
    unittest.main()
