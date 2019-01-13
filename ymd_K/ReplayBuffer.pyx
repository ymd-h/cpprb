# distutils: language = c++

from ymd_K cimport ReplayBuffer

cdef class PyReplayBuffer:
    cdef ReplayBuffer[vector[double],vector[double],double,double] *thisptr
    def __cinit__(self,size):
        print("Replay Buffer")

        self.thisptr = new ReplayBuffer[vector[double],
                                        vector[double],
                                        double,double](size)

    def add(self,observation,action,reward,next_observation,done):
        self.thisptr.add(observation.action,reward,next_observation,done)

    def sample(self,size):
        cdef s = self.thisptr.sample(size)
        return {'obs': get[0,tuple[vector[double],
                                   vector[double],
                                   double,double]](s),
                'act': get[1,tuple[vector[double],
                                   vector[double],
                                   double,double]](s),
                'rew': get[2,tuple[vector[double],
                                   vector[double],
                                   double,double]](s),
                'next_obs': get[3,tuple[vector[double],
                                        vector[double],
                                        double,double]](s),
                'done': get[4,tuple[vector[double],
                                    vector[double],
                                    double,double]](s)}
