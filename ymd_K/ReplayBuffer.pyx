# distutils: language = c++

from ymd_K cimport ReplayBuffer

cdef class PyReplayBuffer:
    cdef ReplayBuffer[vector[double],vector[double],double,double] *thisptr
    def __cinit__(self,size):
        print("Hello World")

        self.thisptr = new ReplayBuffer[vector[double],vector[double],double,double](size)
