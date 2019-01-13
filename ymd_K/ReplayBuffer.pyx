# distutils: language = c++

from ymd_K cimport ReplayBuffer

cdef class PyReplayBuffer:
    cdef ReplayBuffer[vector[double],vector[double],double,double] *thisptr
    cdef vector[vector[double]] *obs
    cdef vector[vector[double]] *act
    cdef vector[double] *rew
    cdef vector[vector[double]] *next_obs
    cdef vector[bool] *done
    def __cinit__(self,size):
        print("Replay Buffer")

        self.thisptr = new ReplayBuffer[vector[double],
                                        vector[double],
                                        double,double](size)
        self.obs = new vector[vector[double]]()
        self.act = new vector[vector[double]]()
        self.rew = new vector[double]()
        self.next_obs = new vector[vector[double]]()
        self.done = new vector[bool]()

    def add(self,observation,action,reward,next_observation,done):
        self.thisptr.add(observation.action,reward,next_observation,done)

    def sample(self,size):
        self.thisptr.sample(size,obs,act,rew,next_obs,done)
        return {'obs': *obs,
                'act': *act,
                'rew': *rew,
                'next_obs': *next_obs,
                'done': *done}
