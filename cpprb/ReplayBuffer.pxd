from libcpp.vector cimport vector
from libcpp cimport bool

cdef extern from "ReplayBuffer.hh" namespace "ymd":
    cdef cppclass CppRingEnvironment[Obs,Act,Rew,Done]:
        CppRingEnvironment(size_t,size_t,size_t)
        void store[O,A,R,NO,D](O*,A*,R*,NO*,D*,size_t)
        void clear()
        size_t get_stored_size()
        void get_buffer_pointers(Obs*&,Act*&,Rew*&,Obs*&,Done*&)
        size_t get_next_index()
        size_t get_buffer_size()
    cdef cppclass CppThreadSafeRingEnvironment[Obs,Act,Rew,Done]:
        CppThreadSafeRingEnvironment(size_t,size_t,size_t,size_t*,size_t*,
                                     Obs*,Act*,Rew*,Obs*,Done*)
        void store[O,A,R,NO,D](O*,A*,R*,NO*,D*,size_t)
        void clear()
        size_t get_stored_size()
        void get_buffer_pointers(Obs*&,Act*&,Rew*&,Obs*&,Done*&)
        size_t get_next_index()
        size_t get_buffer_size()
    cdef cppclass CppSelectiveEnvironment[Obs,Act,Rew,Done]:
        CppSelectiveEnvironment(size_t,size_t,size_t,size_t)
        void store[O,A,R,NO,D](O*,A*,R*,NO*,D*,size_t)
        void clear()
        void get_episode(size_t,size_t&,
                         Obs*&,Act*&,Rew*&,Obs*&,Done*&)
        size_t delete_episode(size_t)
        size_t get_stored_size()
        size_t get_stored_episode_size()
        void get_buffer_pointers(Obs*&,Act*&,Rew*&,Obs*&,Done*&)
        size_t get_next_index()
    cdef cppclass CppPrioritizedSampler[Prio]:
        CppPrioritizedSampler(size_t,Prio)
        void sample(size_t,Prio,vector[Prio]&,vector[size_t]&,size_t)
        void set_priorities(size_t)
        void set_priorities[P](size_t,P)
        void set_priorities(size_t,size_t,size_t)
        void set_priorities[P](size_t,P*,size_t,size_t)
        void update_priorities[I,P](I*,P*,size_t)
        void clear()
        Prio get_max_priority()
    cdef cppclass CppNstepRewardBuffer[Obs,Rew]:
        CppNstepRewardBuffer(size_t,size_t,size_t,Rew)
        void sample[Done](const vector[size_t]&,Rew*,Obs*,Done*)
        void get_buffer_pointers(Rew*,Rew*,Obs*&)
