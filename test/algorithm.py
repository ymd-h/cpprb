import unittest

from cpprb.rl import DQN,RandomPolicy,SoftmaxPolicy

class TestDQN(unittest.TestCase):
    def test_dqn(self):
        class Env_stub:
            def reset(self):
                pass

            def step(self,act):
                pass

        buffer_size = 256
        pre_train = 100
        N_train = 100

        env = Env_stub()
        rp = RandomPolicy()
        sp = SoftmaxPolicy()

        dqn = DQN(env,
                  buffer_size = buffer_size,
                  prioritized = True)

        dqn(rp,pre_train)

        dqn(sp,N_train,
            validation = 5)
