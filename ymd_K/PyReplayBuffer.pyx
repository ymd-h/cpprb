# distutils: language = c++

from libc.stdlib cimport malloc, free
from cython.operator cimport dereference
from cpython cimport PyObject, Py_INCREF
cimport numpy as np
import numpy as np

from ymd_K cimport ReplayBuffer

cdef class VectorWrapper:
    cdef Py_ssize_t *shape
    cdef Py_ssize_t *strides
    cdef Py_ssize_t itemsize
    cdef int ndim
    cdef int value_dim

    def __cinit__(self):
        self.shape   = <Py_ssize_t*>malloc(sizeof(Py_ssize_t) * 2)
        self.strides = <Py_ssize_t*>malloc(sizeof(Py_ssize_t) * 2)

    def __dealloc__(self):
        free(self.shape)
        free(self.strides)

    cdef void update_size(self):
        self.shape[self.ndim -1]   = <Py_ssize_t> self.vec_size()
        self.strides[self.ndim -1] = <Py_ssize_t> self.itemsize

        if self.ndim is 2:
            self.shape[0] = <Py_ssize_t> (self.vec_size()//self.value_dim)
            self.strides[0] = self.value_dim * <Py_ssize_t> self.itemsize

    cdef void set_buffer(self,Py_buffer *buffer):
        pass

    def __getbuffer__(self, Py_buffer *buffer, int flags):
        # relevant documentation http://cython.readthedocs.io/en/latest/src/userguide/buffer.html#a-matrix-class

        self.update_size()

        self.set_buffer(buffer)
        buffer.len = self.vec_size() * self.itemsize
        buffer.readonly = 0
        buffer.ndim = self.ndim
        buffer.shape = self.shape
        buffer.strides = self.strides
        buffer.suboffsets = NULL
        buffer.itemsize = self.itemsize
        buffer.internal = NULL
        buffer.obj = self

    def __releasebuffer__(self, Py_buffer *buffer):
        pass

cdef class VectorIntWrapper(VectorWrapper):
    cdef vector[int] vec

    def __cinit__(self,value_dim=1):
        self.vec = vector[int]()
        self.itemsize = sizeof(int)

        self.ndim = 1 if value_dim is 1 else 2
        self.value_dim = value_dim

    def vec_size(self):
        return self.vec.size()

    cdef void set_buffer(self,Py_buffer* buffer):
        buffer.buf = <void*>(self.vec.data())
        buffer.format = 'i'

    def _push_back(self,v):
        self.vec.push_back(v)

cdef class VectorDoubleWrapper(VectorWrapper):
    cdef vector[double] vec

    def __cinit__(self,value_dim=1):
        self.vec = vector[double]()
        self.itemsize = sizeof(double)

        self.ndim = 1 if value_dim is 1 else 2
        self.value_dim = value_dim

    def vec_size(self):
        return self.vec.size()

    cdef void set_buffer(self,Py_buffer* buffer):
         buffer.buf = <void*>(self.vec.data())
         buffer.format = 'd'

cdef class PyReplayBuffer:
    cdef ReplayBuffer[vector[double],vector[double],double,int] *thisptr
    cdef VectorDoubleWrapper obs
    cdef VectorDoubleWrapper act
    cdef VectorDoubleWrapper rew
    cdef VectorDoubleWrapper next_obs
    cdef VectorIntWrapper done
    def __cinit__(self,size,obs_dim,act_dim):
        print("Replay Buffer")

        self.thisptr = new ReplayBuffer[vector[double],
                                        vector[double],
                                        double,int](size)
        self.obs = VectorDoubleWrapper(obs_dim)
        self.act = VectorDoubleWrapper(act_dim)
        self.rew = VectorDoubleWrapper()
        self.next_obs = VectorDoubleWrapper(obs_dim)
        self.done = VectorIntWrapper()

    def add(self,observation,action,reward,next_observation,done):
        self.thisptr.add(observation,action,reward,next_observation,done)

    def sample(self,size):
        self.thisptr.sample(size,
                            self.obs.vec,
                            self.act.vec,
                            self.rew.vec,
                            self.next_obs.vec,
                            self.done.vec)
        return {'obs': np.asarray(self.obs),
                'act': np.asarray(self.act),
                'rew': np.asarray(self.rew),
                'next_obs': np.asarray(self.next_obs),
                'done': np.asarray(self.done)}
