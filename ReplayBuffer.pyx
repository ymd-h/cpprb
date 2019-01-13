# distutils: language = c++

from ReplayBuffer cimport ReplayBuffer as cppReplayBuffer

cdef class ReplayBuffer:
    def __cinit__(self,size):
        print("Hello World")

        self.rb = cppReplayBuffer(size)

if __name__ is "main":
    rb = ReplayBuffer()
