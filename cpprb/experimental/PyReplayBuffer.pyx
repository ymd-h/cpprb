# distutils: language = c++

cimport numpy as np
import numpy as np
import cython
from cython.operator cimport dereference

from cpprb.ReplayBuffer cimport *


cdef class ReplayBuffer:
    cdef buffer
    cdef size_t buffer_size
    cdef env_dict
    cdef size_t index
    cdef size_t stored_size

    def __cinit__(self,size,env_dict,*args,**kwargs):
        self.env_dict = env_dict
        self.buffer_size = size
        self.stored_size = 0
        self.index = 0

        self.buffer = {}
        for name, defs in self.env_dict.items():
            shape = np.insert(np.asarray(defs.get("shape",1)),0,self.buffer_size)
            self.buffer[name] = np.zeros(shape,dtype=defs.get("dtype",np.double))

    def __init__(self,size,env_dict,*args,**kwargs):
        pass

    def add(self,**kwargs):
        cdef size_t N = np.ravel(kwargs.get("done")).shape[0]

        cdef size_t index = self.index
        cdef int end = index + N
        cdef int remain = 0

        if end > self.buffer_size:
            remain = end - self.buffer_size

        for name, value in kwargs.items():
            value = np.array(value,copy=False,ndmin=2,order='C')
            b = self.buffer[name]

            if end <= self.buffer_size:
                b[index:end] = value
            else:
                b[index:] = value[:-remain]
                b[:remain] = value[-remain:]

        self.stored_size = min(self.stored_size,self.buffer_size)
        self.index = remain or end
        return index

    def _encode_sample(self,idx):
        sample = {}
        for name, defs in self.env_dict.items():
            sample[name] = self.buffer[name][idx]
        return sample

    def sample(self,batch_size):
        cdef idx = np.random.randint(0,self.get_stored_size(),batch_size)
        return self._encode_sample(idx)
