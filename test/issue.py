from multiprocessing import Process
import numpy as np
import unittest

from cpprb import (ReplayBuffer,PrioritizedReplayBuffer,
                   MPReplayBuffer,MPPrioritizedReplayBuffer)
from cpprb import create_buffer


class TestIssue39(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.rb = ReplayBuffer(10,
                              {"obs": {"shape": 3},
                               "act": {"shape": 3},
                               "rew": {},
                               "next_obs": {"shape": 3},
                               "done": {}})
        for i in range(10):
            obs_act = np.array([i for _ in range(3)], dtype=np.float64)
            cls.rb.add(obs=obs_act,
                       act=obs_act,
                       next_obs=obs_act,
                       rew=float(i),
                       done=False)
        cls.s = cls.rb._encode_sample(range(10))

    def test_obs(self):
        self.assertTrue((self.s['obs'] == np.array(((0,0,0),
                                                    (1,1,1),
                                                    (2,2,2),
                                                    (3,3,3),
                                                    (4,4,4),
                                                    (5,5,5),
                                                    (6,6,6),
                                                    (7,7,7),
                                                    (8,8,8),
                                                    (9,9,9)))).all())

    def test_act(self):
        self.assertTrue((self.s['act'] == np.array(((0,0,0),
                                                    (1,1,1),
                                                    (2,2,2),
                                                    (3,3,3),
                                                    (4,4,4),
                                                    (5,5,5),
                                                    (6,6,6),
                                                    (7,7,7),
                                                    (8,8,8),
                                                    (9,9,9)))).all())
    def test_next_obs(self):
        self.assertTrue((self.s['next_obs'] == np.array(((0,0,0),
                                                         (1,1,1),
                                                         (2,2,2),
                                                         (3,3,3),
                                                         (4,4,4),
                                                         (5,5,5),
                                                         (6,6,6),
                                                         (7,7,7),
                                                         (8,8,8),
                                                         (9,9,9)))).all())
    def test_rew(self):
        self.assertTrue((self.s['rew'] == np.array((0,1,2,3,4,
                                                    5,6,7,8,9)).reshape(-1,1)).all())

    def test_done(self):
        self.assertTrue((self.s['done'] == np.zeros(shape=(10))).all())

class TestIssue40(unittest.TestCase):
    def test(self):
        buffer_size = 256
        obs_dim = 3
        act_dim = 1
        rb = ReplayBuffer(buffer_size,{"obs": {"shape": obs_dim}, "act": {"shape": act_dim}, "rew": {}, "next_obs": {"shape": obs_dim}, "done": {}})

        obs = np.ones(shape=(obs_dim))
        act = np.ones(shape=(act_dim))
        rew = 0
        next_obs = np.ones(shape=(obs_dim))
        done = 0

        for i in range(500):
            rb.add(obs=obs,act=act,rew=rew,next_obs=next_obs,done=done)


        batch_size = 32
        sample = rb.sample(batch_size)

class TestIssue43(unittest.TestCase):
    def test_buffer_size(self):
        buffer_size = 1000
        obs_dim = 3
        act_dim = 1

        rb = ReplayBuffer(buffer_size,
                          {"obs": {"shape": obs_dim},
                           "act": {"shape": act_dim},
                           "rew": {},
                           "next_obs": {"shape": obs_dim},
                           "done": {}})
        prb = PrioritizedReplayBuffer(buffer_size,
                                      {"obs": {"shape": obs_dim},
                                       "act": {"shape": act_dim},
                                       "rew": {},
                                       "next_obs": {"shape": obs_dim},
                                       "done": {}})

        self.assertEqual(1000,rb.get_buffer_size())
        self.assertEqual(1000,prb.get_buffer_size())

        rb._encode_sample([i for i in range(1000)])

class TestIssue44(unittest.TestCase):
    def test_cpdef_super(self):
        buffer_size = 256
        obs_dim = 15
        act_dim = 3

        prb = PrioritizedReplayBuffer(buffer_size,
                                      {"obs": {"shape": obs_dim},
                                       "act": {"shape": act_dim},
                                       "rew": {},
                                       "next_obs": {"shape": obs_dim},
                                       "done": {}})

        prb.clear()

class TestIssue45(unittest.TestCase):
    def test_large_size(self):
        buffer_size = 256
        obs_shape = (210, 160, 3)
        act_dim = 4

        rb = create_buffer(buffer_size,obs_shape=obs_shape,act_dim=act_dim,
                           is_discrete_action = True,
                           prioritized = True)

class TestIssue46(unittest.TestCase):
    def test_large_size(self):
        buffer_size = 256
        obs_shape = np.array((210, 160, 3))
        act_dim = 4

        rb = create_buffer(buffer_size,obs_shape=obs_shape,act_dim=act_dim,
                           is_discrete_action = True,
                           prioritized = True)
        rb._encode_sample((0))

class TestIssue90(unittest.TestCase):
    def test_with_empty(self):
        buffer_size = 32
        obs_shape = 3
        act_shape = 4

        rb = ReplayBuffer(buffer_size,{"obs": {"shape": obs_shape},
                                       "act": {"shape": act_shape},
                                       "done": {}})

        tx = rb.get_all_transitions()

        for key in ["obs","act","done"]:
            with self.subTest(key=key):
                self.assertEqual(tx[key].shape[0],0)

    def test_with_one(self):
        buffer_size = 32
        obs_shape = 3
        act_shape = 4

        rb = ReplayBuffer(buffer_size,{"obs": {"shape": obs_shape},
                                       "act": {"shape": act_shape},
                                       "done": {}})

        v = {"obs": np.ones(shape=obs_shape),
             "act": np.zeros(shape=act_shape),
             "done": 0}

        rb.add(**v)

        tx = rb.get_all_transitions()

        for key in ["obs","act","done"]:
            with self.subTest(key=key):
                np.testing.assert_allclose(tx[key],
                                           np.asarray(v[key]).reshape((1,-1)))

class TestIssue61(unittest.TestCase):
    """`ReplayBuffer.add` without "done" key

    Ref: https://gitlab.com/ymd_h/cpprb/-/issues/61

    `ReplayBuffer.add` can accept multiple step without for-loop.
    Inside the member function, step size was taken from "done" key.

    Helper class `StepChecker` is introduced to store one of the keys
    in `self.env_divt` with its "add_shape", and to extract step size
    from `add`ed environment values.
    """

    def test_ReplayBuffer_with_single_step(self):
        buffer_size = 256
        obs_shape = (3,4)
        batch_size = 10

        rb = ReplayBuffer(buffer_size,{"obs": {"shape": obs_shape}})

        v = {"obs": np.ones(shape=obs_shape)}

        rb.add(**v)

        rb.sample(batch_size)

        for _ in range(100):
            rb.add(**v)

        rb.sample(batch_size)

    def test_ReplayBuffer_with_multiple_steps(self):
        buffer_size = 256
        obs_shape = (3,4)
        step_size = 32
        batch_size = 10

        rb = ReplayBuffer(buffer_size,{"obs": {"shape": obs_shape}})

        v = {"obs": np.ones(shape=(step_size,*obs_shape))}

        rb.add(**v)

        rb.sample(batch_size)

        for _ in range(100):
            rb.add(**v)

        rb.sample(batch_size)

    def test_PrioritizedReplayBuffer_with_single_step(self):
        buffer_size = 256
        obs_shape = (3,4)
        batch_size = 10

        rb = PrioritizedReplayBuffer(buffer_size,{"obs": {"shape": obs_shape}})

        v = {"obs": np.ones(shape=obs_shape)}

        rb.add(**v)

        rb.sample(batch_size)

        for _ in range(100):
            rb.add(**v)

        rb.sample(batch_size)

    def test_PrioritizedReplayBuffer_with_multiple_steps(self):
        buffer_size = 256
        obs_shape = (3,4)
        step_size = 32
        batch_size = 10

        rb = PrioritizedReplayBuffer(buffer_size,{"obs": {"shape": obs_shape}})

        v = {"obs": np.ones(shape=(step_size,*obs_shape))}

        rb.add(**v)

        rb.sample(batch_size)

        for _ in range(100):
            rb.add(**v)

        rb.sample(batch_size)

    def test_PrioritizedReplayBuffer_with_single_step_with_priorities(self):
        buffer_size = 256
        obs_shape = (3,4)
        batch_size = 10

        rb = PrioritizedReplayBuffer(buffer_size,{"obs": {"shape": obs_shape}})

        v = {"obs": np.ones(shape=obs_shape),
             "priorities": 0.5}

        rb.add(**v)

        rb.sample(batch_size)

        for _ in range(100):
            rb.add(**v)

        rb.sample(batch_size)

    def test_PrioritizedReplayBuffer_with_multiple_steps_with_priorities(self):
        buffer_size = 256
        obs_shape = (3,4)
        step_size = 32
        batch_size = 10

        rb = PrioritizedReplayBuffer(buffer_size,{"obs": {"shape": obs_shape}})

        v = {"obs": np.ones(shape=(step_size,*obs_shape)),
             "priorities": np.full(shape=(step_size,),fill_value=0.5)}

        rb.add(**v)

        rb.sample(batch_size)

        for _ in range(100):
            rb.add(**v)

        rb.sample(batch_size)

class TestIssue96(unittest.TestCase):
    """
    Bug: PrioritizedReplayBuffer accepted imcompatible shape priority.

    Expected: Raise ValueError

    Ref: https://gitlab.com/ymd_h/cpprb/-/issues/96
    """
    def test_raise_imcompatible_priority_shape(self):
        rb = PrioritizedReplayBuffer(32, env_dict={'a': {'shape': 1}})

        with self.assertRaises(ValueError):
            rb.add(a=np.ones(5), priorities=np.ones(3))

class TestIssue97(unittest.TestCase):
    """
    Bug: stack_compress does not cache non latest transition.

    Expected: Save compress dimension size -1 step transitions as cache

    Ref: https://gitlab.com/ymd_h/cpprb/-/issues/97
    """
    def test_save_cache_with_stack_compress(self):
        rb = PrioritizedReplayBuffer(32, env_dict={'done': {'dtype': 'bool'},
                                                   'a' : {'shape': (3)}},
                                     stack_compress='a')

        a = np.array([0, 1, 2])
        for i in range(3):
            done = i == 2
            rb.add(a=a, done=done)
            if done:
                rb.on_episode_end()
            a += 1
        rb.add(a=np.ones(3), done=False)

        a_ = rb.get_all_transitions()["a"]

        np.testing.assert_allclose(a_,
                                   np.asarray([[0., 1., 2.],
                                               [1., 2., 3.],
                                               [2., 3., 4.],
                                               [1., 1., 1.]]))

class TestIssue108(unittest.TestCase):
    """
    Bug: When "next_of" and "stack_compress" are specified together,
         all the "next_of" cache become last "next_of" item.

    Expected: "next_of" cache should be the next step of original items.

    Ref: https://gitlab.com/ymd_h/cpprb/-/issues/108
    """
    def test_cache_next_of(self):
        stack_size = 3
        episode_len = 5
        rb = ReplayBuffer(32, {"obs": {"shape": (stack_size),"dtype": np.int}},
                          next_of="obs",stack_compress="obs")

        obs = np.arange(episode_len+stack_size+2,dtype=np.int)
        # [0,1,...,episode_len+stack_size+1]
        obs2 = obs + 3*episode_len
        # [3*episode_len,...,4*episode_len+stack_size+1]

        # Add 1st episode
        for i in range(episode_len):
            rb.add(obs=obs[i:i+stack_size],
                   next_obs=obs[i+1:i+1+stack_size])

        s = rb.get_all_transitions()
        self.assertEqual(rb.get_stored_size(),episode_len)
        for i in range(episode_len):
            with self.subTest(i=i):
                np.testing.assert_equal(s["obs"][i],
                                        obs[i:i+stack_size])
                np.testing.assert_equal(s["next_obs"][i],
                                        obs[i+1:i+1+stack_size])

        # Reset environment
        rb.on_episode_end()
        s = rb.get_all_transitions()
        self.assertEqual(rb.get_stored_size(),episode_len)
        for i in range(episode_len):
            with self.subTest(i=i):
                np.testing.assert_equal(s["obs"][i],
                                        obs[i:i+stack_size])
                np.testing.assert_equal(s["next_obs"][i],
                                        obs[i+1:i+1+stack_size])

        # Add 2nd episode
        for i in range(episode_len):
            rb.add(obs=obs2[i:i+stack_size],
                   next_obs=obs2[i+1:i+1+stack_size])

        s = rb.get_all_transitions()
        self.assertEqual(rb.get_stored_size(),2*episode_len)
        for i in range(episode_len):
            with self.subTest(i=i):
                np.testing.assert_equal(s["obs"][i],
                                        obs[i:i+stack_size])
                np.testing.assert_equal(s["next_obs"][i],
                                        obs[i+1:i+1+stack_size])
        for i in range(episode_len):
            with self.subTest(i=i+episode_len):
                np.testing.assert_equal(s["obs"][i+episode_len],
                                        obs2[i:i+stack_size])
                np.testing.assert_equal(s["next_obs"][i+episode_len],
                                        obs2[i+1:i+1+stack_size])

    def test_smaller_episode_than_stack_frame(self):
        """
        `on_episode_end()` caches stack size.

        When episode length is smaller than stack size,
        `on_episode_end()` must avoid caching from previous episode.

        Since cache does not wraparound, this bug does not happen
        at the first episode.

        Ref: https://gitlab.com/ymd_h/cpprb/-/issues/108
        Ref: https://gitlab.com/ymd_h/cpprb/-/issues/110
        """
        stack_size = 4
        episode_len1 = 5
        episode_len2 = 2
        rb = ReplayBuffer(32, {"obs": {"shape": (stack_size),"dtype": np.int}},
                          next_of="obs",stack_compress="obs")

        obs = np.arange(episode_len1+stack_size+2,dtype=np.int)
        obs2= np.arange(episode_len2+stack_size+2,dtype=np.int) + 100

        self.assertEqual(rb.get_current_episode_len(),0)

        # Add 1st episode
        for i in range(episode_len1):
            rb.add(obs=obs[i:i+stack_size],
                   next_obs=obs[i+1:i+1+stack_size])

        s = rb.get_all_transitions()
        self.assertEqual(rb.get_stored_size(),episode_len1)
        self.assertEqual(rb.get_current_episode_len(),episode_len1)
        for i in range(episode_len1):
            with self.subTest(i=i):
                np.testing.assert_equal(s["obs"][i],
                                        obs[i:i+stack_size])
                np.testing.assert_equal(s["next_obs"][i],
                                        obs[i+1:i+1+stack_size])

        # Reset environment
        rb.on_episode_end()
        self.assertEqual(rb.get_current_episode_len(),0)
        s = rb.get_all_transitions()
        self.assertEqual(rb.get_stored_size(),episode_len1)
        for i in range(episode_len1):
            with self.subTest(i=i):
                np.testing.assert_equal(s["obs"][i],
                                        obs[i:i+stack_size])
                np.testing.assert_equal(s["next_obs"][i],
                                        obs[i+1:i+1+stack_size])

        # Add 2nd episode
        for i in range(episode_len2):
            rb.add(obs=obs2[i:i+stack_size],
                   next_obs=obs2[i+1:i+1+stack_size])

        self.assertEqual(rb.get_current_episode_len(),episode_len2)
        s = rb.get_all_transitions()
        self.assertEqual(rb.get_stored_size(),episode_len1 + episode_len2)
        for i in range(episode_len1):
            with self.subTest(i=i,v="obs"):
                np.testing.assert_equal(s["obs"][i],
                                        obs[i:i+stack_size])
            with self.subTest(i=i,v="next_obs"):
                np.testing.assert_equal(s["next_obs"][i],
                                        obs[i+1:i+1+stack_size])
        for i in range(episode_len2):
            with self.subTest(i=i+episode_len1,v="obs"):
                np.testing.assert_equal(s["obs"][i+episode_len1],
                                        obs2[i:i+stack_size])
            with self.subTest(i=i+episode_len1,v="next_obs"):
                np.testing.assert_equal(s["next_obs"][i+episode_len1],
                                        obs2[i+1:i+1+stack_size])


        rb.on_episode_end()
        self.assertEqual(rb.get_current_episode_len(),0)
        s = rb.get_all_transitions()
        self.assertEqual(rb.get_stored_size(),episode_len1 + episode_len2)
        for i in range(episode_len1):
            with self.subTest(i=i,v="obs"):
                np.testing.assert_equal(s["obs"][i],
                                        obs[i:i+stack_size])
            with self.subTest(i=i,v="next_obs"):
                np.testing.assert_equal(s["next_obs"][i],
                                        obs[i+1:i+1+stack_size])
        for i in range(episode_len2):
            with self.subTest(i=i+episode_len1,v="obs"):
                np.testing.assert_equal(s["obs"][i+episode_len1],
                                        obs2[i:i+stack_size])
            with self.subTest(i=i+episode_len1,v="next_obs"):
                np.testing.assert_equal(s["next_obs"][i+episode_len1],
                                        obs2[i+1:i+1+stack_size])


class TestIssue111(unittest.TestCase):
    def test_per_nstep(self):
        """
        PrioritizedReplayBuffer.on_episode_end() ignores Exception

        Ref: https://gitlab.com/ymd_h/cpprb/-/issues/111
        """

        rb = PrioritizedReplayBuffer(32,
                                     {"rew": {}, "done": {}},
                                     Nstep={"size": 4, "rew": "rew", "gamma": 0.5})

        for _ in range(10):
            rb.add(rew=0.5,done=0.0)

        rb.add(rew=0.5,done=1.0)
        rb.on_episode_end()

        s = rb.sample(16)

        self.assertIn("discounts",s)

class TestIssue112(unittest.TestCase):
    """
    dtype is converted to integer type

    Ref: https://gitlab.com/ymd_h/cpprb/-/issues/112

    Even set np.bool, it become np.int64.
    This bug was introduced when fixing https://gitlab.com/ymd_h/cpprb/-/issues/105
    """
    def test_dtype_check(self):
        types = [np.bool_,
                 np.bool8,
                 np.byte,
                 np.short,
                 np.intc,
                 np.int_,
                 np.longlong,
                 np.intp,
                 np.int8,
                 np.int16,
                 np.int32,
                 np.int64,
                 np.ubyte,
                 np.ushort,
                 np.uintc,
                 np.uint,
                 np.ulonglong,
                 np.uintp,
                 np.uint8,
                 np.uint16,
                 np.uint32,
                 np.uint64,
                 np.half,
                 np.single,
                 np.double,
                 np.float_,
                 np.longfloat,
                 np.float16,
                 np.float32,
                 np.float64,
                 np.csingle,
                 np.complex_,
                 np.clongfloat,
                 np.complex64,
                 np.complex128]

        for d in types:
            with self.subTest(type=d):
                b = ReplayBuffer(10,{"a": {"dtype": d}})
                b.add(a=np.ones(1,dtype=d))
                self.assertEqual(b.get_all_transitions()["a"].dtype,d)

    def test_python_type(self):
        types = [bool,
                 int,
                 float]

        for d in types:
            with self.subTest(type=d):
                b = ReplayBuffer(10,{"a": {"dtype": d}})
                b.add(a=d(1))
                self.assertEqual(b.get_all_transitions()["a"].dtype,d)


class TestIssue114(unittest.TestCase):
    """
    ReplayBuffer.sample raises IndexError

    Cache mechanism with next_of and compress_any seem to be broken

    Ref: https://gitlab.com/ymd_h/cpprb/-/issues/114
    """
    def test_has_next_of(self):
        bsize = 10
        rb = ReplayBuffer(bsize,{"a": {}},next_of="a")
        a = np.random.rand(bsize + 1)

        for i in range(bsize):
            rb.add(a=a[i],next_a=a[i+1])

        _next_a = np.ravel(rb.get_all_transitions()["next_a"])
        np.testing.assert_allclose(_next_a,a[1:bsize+1])

        for i in range(bsize):
            rb._encode_sample([i])

        rb.clear()

        for i in range(bsize):
            rb.add(a=a[i],next_a=a[i+1])
            rb.on_episode_end()

        _next_a = np.ravel(rb.get_all_transitions()["next_a"])
        np.testing.assert_allclose(_next_a,a[1:bsize+1])

        for i in range(bsize):
            rb._encode_sample([i])

    def test_stack_compress(self):
        bsize = 10
        odim = 2
        ssize = 2
        rb = ReplayBuffer(bsize,{"a": {"shape": (odim,ssize)}},stack_compress="a")
        a = np.random.rand(odim,bsize + ssize-1)

        for i in range(bsize):
            rb.add(a=a[:,i:i+ssize])

        _a = rb.get_all_transitions()["a"]
        for i in range(bsize):
            with self.subTest(i=i,label="without cache"):
                np.testing.assert_allclose(_a[i],a[:,i:i+ssize])

        for i in range(bsize):
            rb._encode_sample([i])

        rb.clear()

        for i in range(bsize):
            rb.add(a=a[:,i:i+ssize])
            rb.on_episode_end()

        _a = rb.get_all_transitions()["a"]
        for i in range(bsize):
            with self.subTest(i=i,label="without cache"):
                np.testing.assert_allclose(_a[i],a[:,i:i+ssize])

        for i in range(bsize):
            rb._encode_sample([i])

class TestIssue116(unittest.TestCase):
    """
    Raise KeyError: 'discounts' for Nstep

    Ref: https://gitlab.com/ymd_h/cpprb/-/issues/116
    """
    def test_after_nstep(self):
        buffer_size = 100

        obs_shape = 4
        env_dict={"state": {"shape": obs_shape},
                  "next_state": {"shape": obs_shape},
                  "reward": {},
                  "done": {}}

        def make_data_dict(n, nstep=None):
            data={"state": np.zeros((n, obs_shape)),
                  "reward": np.zeros(n),
                  "next_state": np.zeros((n, obs_shape)),
                  "done": np.zeros(n)}
            return data

        ReplayBuffer(buffer_size, env_dict=env_dict,
                     Nstep={"size": 4, "gamma": 0.99,
                            "rew": "reward",
                            "next": ["next_state"]})

        rbc = ReplayBuffer(buffer_size, env_dict=env_dict)
        data_dict = make_data_dict(32)
        rbc.add(**data_dict)

class TestIssue128(unittest.TestCase):
    """
    Read only indexes and priorities raise ValueError

    Ref: https://gitlab.com/ymd_h/cpprb/-/issues/128
    """
    def test_read_only_priority(self):
        buffer_size = 100
        batch_size = 32

        env_dict = {"done": {}}

        done = np.zeros(2)
        ps = np.ones_like(done)
        ps.setflags(write=False)

        rb = PrioritizedReplayBuffer(buffer_size,env_dict)
        rb.add(done=done,priority=ps)

        sample = rb.sample(batch_size)
        ps2 = sample["weights"]
        ps2.setflags(write=False)

        rb.update_priorities(sample["indexes"],ps2)

    def test_read_only_mppriority(self):
        buffer_size = 100
        batch_size = 32

        env_dict = {"done": {}}

        done = np.zeros(2)
        ps = np.ones_like(done)
        ps.setflags(write=False)

        rb = MPPrioritizedReplayBuffer(buffer_size,env_dict)
        rb.add(done=done,priority=ps)

        sample = rb.sample(batch_size)
        ps2 = sample["weights"]
        ps2.setflags(write=False)

        rb.update_priorities(sample["indexes"],ps2)

def add_args(rb,args):
    for arg in args:
        rb.add(**arg)

class TestIssue130(unittest.TestCase):
    """
    MPPrioritizedReplayBuffer cannot use np.float16

    `NotImplementedError: Converting dtype('float16') to a ctypes type`

    Ref: https://gitlab.com/ymd_h/cpprb/-/issues/130
    """
    def test_np_float16(self):
        buffer_size = 256
        rb = MPReplayBuffer(buffer_size,{"done": {"dtype": np.float16}})

        self.assertEqual(rb.get_next_index(),0)
        self.assertEqual(rb.get_stored_size(),0)

        p = Process(target=add_args,args=[rb, [{"done": i} for i in range(100)]])
        p.start()
        p.join()

        self.assertEqual(rb.get_next_index(),100)
        self.assertEqual(rb.get_stored_size(),100)

        done = rb.get_all_transitions()["done"]

        np.testing.assert_allclose(done.ravel(),
                                   np.arange(100,dtype=np.float16))

        self.assertEqual(done.dtype, np.float16)

    def test_dtypes(self):
        buffer_size = 1
        for d in [np.byte, np.ubyte, np.short, np.ushort, np.intc, np.uintc,
                  np.int_, np.uint, np.longlong, np.ulonglong,
                  np.half, np.double, np.longdouble,
                  np.int8, np.int16, np.int32, np.int64,
                  np.uint8, np.uint16, np.uint32, np.uint64, np.intp,
                  np.float16, np.float32, np.float64]:

            if np.dtype(d).itemsize > 8:
                # 128bit is not supported
                continue

            with self.subTest(dtype=d):
                rb = MPReplayBuffer(buffer_size,{"a": {"dtype": d}})
                rb.add(a=np.zeros((1,),dtype=d))
                a = rb.get_all_transitions()["a"].ravel()
                self.assertEqual(a.dtype, d)
                np.testing.assert_allclose(a,np.zeros((1,),dtype=d))


if __name__ == '__main__':
    unittest.main()
