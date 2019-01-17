from libcpp.vector cimport vector
from libcpp cimport bool

cdef extern from "ReplayBuffer.hh" namespace "ymd":
  cdef cppclass ReplayBuffer[T1,T2,T3,T4]:
    ReplayBuffer(size_t)
    void add(T1,T2,T3,T1,T4)
    void sample(size_t,
                vector[T1a]&,
                vector[T2a]&,
                vector[T3a]&,
                vector[T1a]&,
                vector[T4a]&)
