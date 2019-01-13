# distutils: language = c++

from cython.operator cimport dereference
from ymd_K cimport ReplayBuffer

cdef class PyReplayBuffer:
    cdef ReplayBuffer[vector[double],vector[double],double,bool] *thisptr
    cdef vector[vector[double]] *obs
    cdef vector[vector[double]] *act
    cdef vector[double] *rew
    cdef vector[vector[double]] *next_obs
    cdef vector[bool] *done
    def __cinit__(self,size):
        print("Replay Buffer")

        self.thisptr = new ReplayBuffer[vector[double],
                                        vector[double],
                                        double,bool](size)
        self.obs = new vector[vector[double]]()
        self.act = new vector[vector[double]]()
        self.rew = new vector[double]()
        self.next_obs = new vector[vector[double]]()
        self.done = new vector[bool]()

    def add(self,observation,action,reward,next_observation,done):
        self.thisptr.add(observation,action,reward,next_observation,done)

    def sample(self,size):
        self.thisptr.sample(size,
                            dereference(self.obs),
                            dereference(self.act),
                            dereference(self.rew),
                            dereference(self.next_obs),
                            dereference(self.done))
        return {'obs': dereference(self.obs),
                'act': dereference(self.act),
                'rew': dereference(self.rew),
                'next_obs': dereference(self.next_obs),
                'done': dereference(self.done)}
