# distutils: language = c++

from cython.operator cimport dereference
from cpython cimport PyObject, Py_INCREF
cimport numpy as np
import numpy as np

from ymd_K cimport ReplayBuffer

cdef class VectorWrapper[T]:
    cdef vector[T] vec
    cdef Py_ssize_t shape[1]
    cdef Py_ssize_t strides[1]

    cdef format_type(int _):
        return 'i'

    cdef format_type(double _):
        return 'f'

    cdef format_type(float _):
        return 'f'

    def __cinit__(self):
        vec = vector[T]()

    def __getbuffer__(self, Py_buffer *buffer, int flags):
        # relevant documentation http://cython.readthedocs.io/en/latest/src/userguide/buffer.html#a-matrix-class
        cdef Py_ssize_t itemsize = sizeof(self.vec[0])

        self.shape[0] = self.vec.size()
        self.strides[0] = sizeof(T)
        buffer.buf = <char *>&(self.vec[0])
        buffer.format = self.format_type(<T>( 1 )) # float or int
        buffer.internal = NULL
        buffer.itemsize = itemsize
        buffer.len = self.v.size() * itemsize   # product(shape) * itemsize
        buffer.ndim = 1
        buffer.obj = self
        buffer.readonly = 0
        buffer.shape = self.shape
        buffer.strides = self.strides
        buffer.suboffsets = NULL

    def __releasebuffer__(self, Py_buffer *buffer):
        pass

cdef class PyReplayBuffer:
    cdef ReplayBuffer[vector[double],vector[double],double,int] *thisptr
    cdef vector[vector[double]] *obs
    cdef vector[vector[double]] *act
    cdef VectorWrapper[double] rew
    cdef vector[vector[double]] *next_obs
    cdef VectorWrapper[int] done
    def __cinit__(self,size):
        print("Replay Buffer")

        self.thisptr = new ReplayBuffer[vector[double],
                                        vector[double],
                                        double,int](size)
        self.obs = new vector[vector[double]]()
        self.act = new vector[vector[double]]()
        self.rew = VectorWrapper[double]()
        self.next_obs = new vector[vector[double]]()
        self.done = VectorWrapper[int]()

    def add(self,observation,action,reward,next_observation,done):
        self.thisptr.add(observation,action,reward,next_observation,done)

    def sample(self,size):
        self.thisptr.sample(size,
                            dereference(self.obs),
                            dereference(self.act),
                            self.rew.vec,
                            dereference(self.next_obs),
                            self.done.vec)
        return {'obs': dereference(self.obs),
                'act': dereference(self.act),
                'rew': np.asarray(self.rew),
                'next_obs': dereference(self.next_obs),
                'done': np.asarray(self.done)}
