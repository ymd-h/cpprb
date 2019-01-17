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

    def __cinit__(self):
        vec = vector[T]()

    def __getbuffer__(self, Py_buffer *buffer, int flags):
        # relevant documentation http://cython.readthedocs.io/en/latest/src/userguide/buffer.html#a-matrix-class
        cdef Py_ssize_t itemsize = sizeof(self.vec[0])

        self.shape[0] = self.vec.size()
        self.strides[0] = sizeof(T)
        buffer.buf = <char *>&(self.vec[0])
        buffer.format = 'f' # float
        buffer.internal = NULL
        buffer.itemsize = itemsize
        buffer.len = self.v.size() * itemsize   # product(shape) * itemsize
        buffer.ndim = 1
        buffer.obj = self
        buffer.readonly = 0
        buffer.shape = self.shape
        buffer.strides = self.strides
        buffer.suboffsets = NULL

cdef class PyReplayBuffer:
    cdef ReplayBuffer[vector[double],vector[double],double,int] *thisptr
    cdef vector[vector[double]] *obs
    cdef vector[vector[double]] *act
    cdef vector[double] *rew
    cdef vector[vector[double]] *next_obs
    cdef vector[int] *done
    def __cinit__(self,size):
        print("Replay Buffer")

        self.thisptr = new ReplayBuffer[vector[double],
                                        vector[double],
                                        double,int](size)
        self.obs = new vector[vector[double]]()
        self.act = new vector[vector[double]]()
        self.rew = new vector[double]()
        self.next_obs = new vector[vector[double]]()
        self.done = new vector[int]()

    def add(self,observation,action,reward,next_observation,done):
        self.thisptr.add(observation,action,reward,next_observation,done)

    def sample(self,size):
        self.thisptr.sample(size,
                            dereference(self.obs),
                            dereference(self.act),
                            dereference(self.rew),
                            dereference(self.next_obs),
                            dereference(self.done))
        return {'obs': dereference(self.obs),
                'act': dereference(self.act),
                'rew': dereference(self.rew),
                'next_obs': dereference(self.next_obs),
                'done': dereference(self.done)}
