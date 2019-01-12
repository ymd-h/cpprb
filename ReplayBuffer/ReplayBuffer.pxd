from libcpp.vector cimport vector
from libcpp.tuple cimport tuple

cdef extern from "ReplayBuffer.hh" namespace "ymd":
  cdef cppclass ReplayBuffer:
    ReplayBuffer(std::size_t)
    void add(vector[double],vector[double],double,vector[double],bool)
    tuple[vector[vector[double]],vector[vector[double]],vector[double],vector[vector],vecto[bool]] sample(size_t)
