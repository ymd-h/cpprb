# distutils: language = c++

from cython.operator cimport dereference
from cpython cimport PyObject, Py_INCREF
cimport numpy as np
import numpy as np

from ymd_K cimport ReplayBuffer

cdef class VectorWrapper:
    cdef Py_ssize_t itemsize
    cdef Py_buffer buffer

    def vec_size(self):
        pass

    cdef char* vec_addr(self):
        pass

    cdef void update_buffer(self):
        self.buffer.shape = [<Py_ssize_t> self.vec_size()]
        self.buffer.strides = [<Py_ssize_t> self.itemsize]
        self.buffer.ndim = 1

    def __getbuffer__(self, Py_buffer *buffer, int flags):
        # relevant documentation http://cython.readthedocs.io/en/latest/src/userguide/buffer.html#a-matrix-class

        self.update_buffer()
        buffer.buf = self.vec_addr()
        buffer.format = self.format_type # float or int
        buffer.internal = NULL
        buffer.itemsize = self.itemsize
        buffer.len = self.vec_size() * self.itemsize   # product(shape) * itemsize
        buffer.ndim = self.buffer.ndim
        buffer.obj = self
        buffer.readonly = 0
        buffer.shape = self.buffer.shape
        buffer.strides = self.buffer.strides
        buffer.suboffsets = NULL

    def __releasebuffer__(self, Py_buffer *buffer):
        pass

cdef class VectorWrapperInt(VectorWrapper):
   cdef vector[int] vec
   format_type = "i"

   def __cinit__(self):
       self.vec = vector[int]()
       self.itemsize = sizeof(int)

   def vec_size(self):
       return self.vec.size()

   cdef char* vec_addr(self):
       return <char*>(self.vec.data())

cdef class VectorWrapperDouble(VectorWrapper):
   cdef vector[double] vec
   format_type = "d"

   def __cinit__(self):
       self.vec = vector[double]()
       self.itemsize = sizeof(double)

   def vec_size(self):
       return self.vec.size()

   cdef char* vec_addr(self):
       return <char*>(self.vec.data())

cdef class VectorWrapperDouble2d(VectorWrapperDouble):
    cdef Py_ssize_t ndim
    def __cinit__(self,ndim=2):
        self.ndim = ndim

    cdef void update_buffer(self):
        self.buffer.shape = [<Py_ssize_t> (self.vec_size()//self.ndim),self.ndim]
        self.buffer.strides = [self.ndim * <Py_ssize_t> self.itemsize,
                               <Py_ssize_t> self.itemsize]
        self.buffer.ndim = 2

cdef class PyReplayBuffer:
    cdef ReplayBuffer[vector[double],vector[double],double,int] *thisptr
    cdef VectorWrapperDouble2d obs
    cdef VectorWrapperDouble2d act
    cdef VectorWrapperDouble rew
    cdef VectorWrapperDouble2d next_obs
    cdef VectorWrapperInt done
    def __cinit__(self,size,obs_dim,act_dim):
        print("Replay Buffer")

        self.thisptr = new ReplayBuffer[vector[double],
                                        vector[double],
                                        double,int](size)
        self.obs = VectorWrapperDouble2d(obs_dim)
        self.act = VectorWrapperDouble2d(act_dim)
        self.rew = VectorWrapperDouble()
        self.next_obs = VectorWrapperDouble2d(obs_dim)
        self.done = VectorWrapperInt()

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
