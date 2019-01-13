from libcpp.vector cimport vector
from libcpp cimport bool

cdef extern from "<tuple>" namespace "std" nogil:
  cdef cppclass tuple[T1,T2,T3,T4,T5]:
    tuple(T1,T2,T3,T4,T5) except +


cdef extern from "ReplayBuffer.hh" namespace "ymd":
  cdef cppclass ReplayBuffer[T1,T2,T3,T4]:
    ReplayBuffer(size_t)
    void add(T1,T2,T3,T1,T4)
    tuple[T1,T2,T3,T1,T4] sample(size_t)
