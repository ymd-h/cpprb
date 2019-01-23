from libcpp.vector cimport vector
from libcpp cimport bool

cdef extern from "ReplayBuffer.hh" namespace "ymd":
    cdef cppclass ReplayBuffer[T1,T2,T3,T4]:
        ReplayBuffer(size_t,size_t,size_t)
        void add(T1*,T2*,T3*,T1*,T4*,size_t)
        void sample(size_t,
                    vector[T1]&,
                    vector[T2]&,
                    vector[T3]&,
                    vector[T1]&,
                    vector[T4]&)
    cdef cppclass PrioritizedReplayBuffer[T1,T2,T3,T4,T5]:
        PrioritizedReplayBuffer(size_t,size_t,size_t,T5)
        void add(T1*,T2*,T3*,T1*,T4*,size_t)
        void sample(size_t,T5,
                    vector[T1]&,
                    vector[T2]&,
                    vector[T3]&,
                    vector[T1]&,
                    vector[T4]&,
                    vector[T5]&,
                    vector[size_t]&)
        void update_priorities(vector[size_t]&,vector[T5]&)
