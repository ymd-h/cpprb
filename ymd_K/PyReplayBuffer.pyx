# distutils: language = c++

from cython.operator cimport dereference
from cpython cimport PyObject, Py_INCREF
cimport numpy as np
import numpy as np

from ymd_K cimport ReplayBuffer

cdef class VectorWrapper:
    cdef Py_ssize_t shape[1]
    cdef Py_ssize_t strides[1]
    cdef Py_ssize_t itemsize

    def vec_size(self):
        pass

    cdef char* vec_addr(self):
        pass

    def __getbuffer__(self, Py_buffer *buffer, int flags):
        # relevant documentation http://cython.readthedocs.io/en/latest/src/userguide/buffer.html#a-matrix-class

        self.shape[0] = self.vec_size()
        self.strides[0] = self.itemsize
        buffer.buf = self.vec_addr()
        buffer.format = self.format_type # float or int
        buffer.internal = NULL
        buffer.itemsize = self.itemsize
        buffer.len = self.vec_size() * self.itemsize   # product(shape) * itemsize
        buffer.ndim = 1
        buffer.obj = self
        buffer.readonly = 0
        buffer.shape = self.shape
        buffer.strides = self.strides
        buffer.suboffsets = NULL

    def __releasebuffer__(self, Py_buffer *buffer):
        pass

cdef class VectorWrapperInt(VectorWrapper):
   cdef vector[int] vec
   format_type = 'i'

   def __cinit__(self):
       self.vec = vector[int]()
       self.itemsize = sizeof(int)

   def vec_size(self):
       return self.vec.size()

   cdef char* vec_addr(self):
       return <char*>(self.vec.data())

cdef class VectorWrapperDouble(VectorWrapper):
   cdef vector[double] vec
   format_type = 'f'

   def __cinit__(self):
       self.vec = vector[double]()
       self.itemsize = sizeof(double)

   def vec_size(self):
       return self.vec.size()

   cdef char* vec_addr(self):
       return <char*>(self.vec.data())

cdef class PyReplayBuffer:
    cdef ReplayBuffer[vector[double],vector[double],double,int] *thisptr
    cdef vector[vector[double]] *obs
    cdef vector[vector[double]] *act
    cdef VectorWrapperDouble rew
    cdef vector[vector[double]] *next_obs
    cdef VectorWrapperInt done
    def __cinit__(self,size):
        print("Replay Buffer")

        self.thisptr = new ReplayBuffer[vector[double],
                                        vector[double],
                                        double,int](size)
        self.obs = new vector[vector[double]]()
        self.act = new vector[vector[double]]()
        self.rew = VectorWrapperDouble()
        self.next_obs = new vector[vector[double]]()
        self.done = VectorWrapperInt()

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
