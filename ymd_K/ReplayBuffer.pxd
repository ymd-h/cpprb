from libcpp.vector cimport vector
from libcpp cimport bool

cdef extern from "ReplayBuffer.hh" namespace "ymd":
    cdef cppclass ReplayBuffer[Obs,Act,Rew,Done]:
        ReplayBuffer(size_t,size_t,size_t)
        void add(Obs*,Act*,Rew*,Obs*,Done*,size_t)
        void clear()
        size_t get_stored_size()
        void get_buffer_pointers(Obs*&,Act*&,Rew*&,Obs*&,Done*&)
    cdef cppclass PrioritizedReplayBuffer[Obs,Act,Rew,Done,Prio]:
        PrioritizedReplayBuffer(size_t,size_t,size_t,Prio)
        PrioritizedReplayBuffer(ReplayBuffer[Obs,Act,Rew,Done]*,Prio)
        void add(Obs*,Act*,Rew*,Obs*,Done*,size_t)
        void add(Obs*,Act*,Rew*,Obs*,Done*,Prio*,size_t)
        void add(Obs*,Act*,Rew*,Obs*,Done*)
        void add(Obs*,Act*,Rew*,Obs*,Done*,Prio)
        void prioritized_indexes(size_t,Prio,vector[Prio]&,vector[size_t]&)
        void update_priorities(vector[size_t]&,vector[Prio]&)
        void clear()
        size_t get_stored_size()
        Prio get_max_priority()
        void get_buffer_pointers(Obs*&,Act*&,Rew*&,Obs*&,Done*&)
