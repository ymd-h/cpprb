from libcpp.vector cimport vector
from libcpp cimport bool

cdef extern from "ReplayBuffer.hh" namespace "ymd":
    void clear[B](B*)
    size_t get_buffer_size[B](B*)
    size_t get_stored_size[B](B*)
    size_t get_next_index[B](B*)

    cdef cppclass CppSelectiveEnvironment[Obs,Act,Rew,Done]:
        CppSelectiveEnvironment(size_t,size_t,size_t,size_t,size_t) except +
        size_t store[O,A,R,NO,D](O*,A*,R*,NO*,D*,size_t)
        void get_episode(size_t,size_t&,
                         Obs*&,Act*&,Rew*&,Obs*&,Done*&)
        size_t delete_episode(size_t)
        size_t get_stored_episode_size()
        void get_buffer_pointers(Obs*&,Act*&,Rew*&,Obs*&,Done*&)
    cdef cppclass CppPrioritizedSampler[Prio]:
        CppPrioritizedSampler(size_t,Prio) except +
        void sample(size_t,Prio,vector[Prio]&,vector[size_t]&,size_t)
        void set_priorities(size_t)
        void set_priorities[P](size_t,P)
        void set_priorities(size_t,size_t,size_t)
        void set_priorities[P](size_t,P*,size_t,size_t)
        void update_priorities[I,P](I*,P*,size_t)
        Prio get_max_priority()
        void set_eps(Prio)
    cdef cppclass CppThreadSafePrioritizedSampler[Prio]:
        CppThreadSafePrioritizedSampler(size_t,Prio,Prio*,
                                        Prio*,bool*,bool*,Prio*,bool*,bool*,
                                        bool,float) except +
        void sample(size_t,Prio,vector[Prio]&,vector[size_t]&,size_t)
        void set_priorities(size_t)
        void set_priorities[P](size_t,P)
        void set_priorities(size_t,size_t,size_t)
        void set_priorities[P](size_t,P*,size_t,size_t)
        void update_priorities[I,P](I*,P*,size_t)
        Prio get_max_priority()
        void weak_update_changed()
