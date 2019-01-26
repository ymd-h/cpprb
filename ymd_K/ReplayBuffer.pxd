from libcpp.vector cimport vector
from libcpp cimport bool

cdef extern from "ReplayBuffer.hh" namespace "ymd":
    cdef cppclass ReplayBuffer[Obs,Act,Rew,Done]:
        ReplayBuffer(size_t,size_t,size_t)
        void add(Obs*,Act*,Rew*,Obs*,Done*,size_t)
        void sample(size_t,
                    Obs*,
                    Act*,
                    Rew*,
                    Obs*,
                    Done*)
    cdef cppclass PrioritizedReplayBuffer[Obs,Act,Rew,Done,Prio]:
        PrioritizedReplayBuffer(size_t,size_t,size_t,Prio)
        void add(Obs*,Act*,Rew*,Obs*,Done*,size_t)
        void add(Obs*,Act*,Rew*,Obs*,Done*)
        void sample(size_t,Prio,
                    Obs*,
                    Act*,
                    Rew*,
                    Obs*,
                    Done*,
                    vector[Prio]&,
                    vector[size_t]&)
        void update_priorities(vector[size_t]&,vector[Prio]&)
