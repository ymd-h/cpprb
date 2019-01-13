# distutils: language = c++

from ReplayBuffer cimport ReplayBuffer

cdef class PyReplayBuffer:
    cdef ReplayBuffer *thisptr
    def __cinit__(self,size):
        print("Hello World")

        self.thisptr = new ReplayBuffer(size)
