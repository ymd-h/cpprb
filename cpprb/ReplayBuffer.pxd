from libcpp.vector cimport vector
from libcpp cimport bool

cdef extern from "ReplayBuffer.hh" namespace "ymd":
    cdef cppclass RingEnvironment[Obs,Act,Rew,Done]:
        RingEnvironment(size_t,size_t,size_t)
        void store(Obs*,Act*,Rew*,Obs*,Done*,size_t)
        void clear()
        size_t get_stored_size()
        void get_buffer_pointers(Obs*&,Act*&,Rew*&,Obs*&,Done*&)
        size_t get_next_index()
    cdef cppclass PrioritizedSampler[Prio]:
        PrioritizedSampler(size_t,Prio)
        void sample(size_t,Prio,vector[Prio]&,vector[size_t]&,size_t)
        void set_priorities(size_t)
        void set_priorities(size_t,Prio)
        void set_priorities(size_t,size_t,size_t)
        void set_priorities(size_t,Prio*,size_t,size_t)
        void update_priorities(vector[size_t]&,vector[Prio]&)
        void clear()
        Prio get_max_priority()
    cdef cppclass NstepRewardBuffer[Obs,Rew]:
        NstepRewardBuffer(size_t,size_t,size_t,Rew)
        void sample[Done](const vector[size_t]&,Rew*,Obs*,Done*)
        void get_buffer_pointers(Rew*,Rew*,Obs*&)
