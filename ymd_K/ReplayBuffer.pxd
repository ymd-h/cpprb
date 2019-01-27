from libcpp.vector cimport vector
from libcpp cimport bool

cdef extern from "ReplayBuffer.hh" namespace "ymd":
    cdef cppclass InternalBuffer[Obs,Act,Rew,Done]:
        InternalBuffer(size_t,size_t,size_t)
        void store(Obs*,Act*,Rew*,Obs*,Done*,size_t)
        void clear()
        size_t get_stored_size()
        void get_buffer_pointers(Obs*&,Act*&,Rew*&,Obs*&,Done*&)
    cdef cppclass PrioritizedSampler[size_t,Prio]:
        PrioritizedSampler(Prio)
        void set_priorities(size_t)
        void set_priorities(size_t,Prio)
        void set_priorities(size_t,size_t,size_t)
        void set_priorities(size_t,Prio*,size_t,size_t)
        void update_priorities(vector[size_t]&,vector[Prio]&)
        void clear()
        Prio get_max_priority()
