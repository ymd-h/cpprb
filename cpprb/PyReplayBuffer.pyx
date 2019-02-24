# distutils: language = c++

from multiprocessing.sharedctypes import RawArray
import ctypes
cimport numpy as np
import numpy as np
import cython

from cpprb cimport ReplayBuffer

from .VectorWrapper cimport *
from .VectorWrapper import (VectorInt,VectorDouble,VectorSize_t,PointerDouble)

Obs      = cython.fused_type(cython.float, cython.double)
Act      = cython.fused_type(cython.float, cython.double)
Rew      = cython.fused_type(cython.float, cython.double)
Next_Obs = cython.fused_type(cython.float, cython.double)
Done     = cython.fused_type(cython.float, cython.double)
Prio     = cython.fused_type(cython.float, cython.double)

cdef class Environment:
    cdef PointerDouble obs
    cdef PointerDouble act
    cdef PointerDouble rew
    cdef PointerDouble next_obs
    cdef PointerDouble done
    cdef int buffer_size
    cdef int obs_dim
    cdef int act_dim

    def __cinit__(self,size,obs_dim,act_dim,**kwargs):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.obs = PointerDouble(ndim=2,value_dim=obs_dim,size=size)
        self.act = PointerDouble(ndim=2,value_dim=act_dim,size=size)
        self.rew = PointerDouble(ndim=2,value_dim=1,size=size)
        self.next_obs = PointerDouble(ndim=2,value_dim=obs_dim,size=size)
        self.done = PointerDouble(ndim=2,value_dim=1,size=size)

    def add(self,obs,act,rew,next_obs,done):
        if obs.ndim == 1:
            self._add_1(obs,act,rew,next_obs,done)
        else:
            self._add_N(obs,act,rew,next_obs,done,obs.shape[0])

    def _encode_sample(self,idx):
        return {'obs': np.asarray(self.obs)[idx],
                'act': np.asarray(self.act)[idx],
                'rew': np.asarray(self.rew)[idx],
                'next_obs': np.asarray(self.next_obs)[idx],
                'done': np.asarray(self.done)[idx]}

    def get_buffer_size(self):
        return self.buffer_size

cdef class RingEnvironment(Environment):
    cdef CppRingEnvironment[double,double,double,double] *buffer
    def __cinit__(self,size,obs_dim,act_dim,**kwargs):
        self.buffer = new CppRingEnvironment[double,double,double,double](size,
                                                                          obs_dim,
                                                                          act_dim)

        self.buffer.get_buffer_pointers(self.obs.ptr,
                                        self.act.ptr,
                                        self.rew.ptr,
                                        self.next_obs.ptr,
                                        self.done.ptr)

        self.buffer_size = self.buffer.get_buffer_size()

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def _add_N(self,
               np.ndarray[Obs     ,ndim = 2, mode="c"] obs not None,
               np.ndarray[Act     ,ndim = 2, mode="c"] act not None,
               np.ndarray[Rew     ,ndim = 1, mode="c"] rew not None,
               np.ndarray[Next_Obs,ndim = 2, mode="c"] next_obs not None,
               np.ndarray[Done    ,ndim = 1, mode="c"] done not None,
               size_t N=1):
        return self.buffer.store(&obs[0,0],&act[0,0],&rew[0],
                                 &next_obs[0,0],&done[0],N)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def _add_1(self,
               np.ndarray[Obs     ,ndim = 1, mode="c"] obs not None,
               np.ndarray[Act     ,ndim = 1, mode="c"] act not None,
               Rew rew,
               np.ndarray[Next_Obs,ndim = 1, mode="c"] next_obs not None,
               double done):
        return self.buffer.store(&obs[0],&act[0],&rew,&next_obs[0],&done,1)

    def clear(self):
        return self.buffer.clear()

    def get_stored_size(self):
        return self.buffer.get_stored_size()

    def get_next_index(self):
        return self.buffer.get_next_index()

cdef class ProcessSharedRingEnvironment(Environment):
    cdef CppThreadSafeRingEnvironment[double,double,double,double] *buffer
    cdef stored_size_v
    cdef next_index_v
    cdef obs_v
    cdef act_v
    cdef rew_v
    cdef next_obs_v
    cdef done_v
    def __cinit__(self,size,obs_dim,act_dim,*,
                  stored_size=None,next_index=None,
                  obs=None,act=None,rew=None,next_obs=None,done=None,
                  **kwargs):
        self.stored_size_v = stored_size or RawArray(ctypes.c_size_t,1)
        self.next_index_v = next_index or RawArray(ctypes.c_size_t,1)
        self.obs_v = obs or RawArray(ctypes.c_double,size*obs_dim)
        self.act_v = act or RawArray(ctypes.c_double,size*act_dim)
        self.rew_v = rew or RawArray(ctypes.c_double,size)
        self.next_obs_v = next_obs or RawArray(ctypes.c_double,size*obs_dim)
        self.done_v = done or RawArray(ctypes.c_double,size)

        cdef size_t [:] stored_size_view = self.stored_size_v
        cdef size_t [:] next_index_view = self.next_index_v
        cdef double [:] obs_view = self.obs_v
        cdef double [:] act_view = self.act_v
        cdef double [:] rew_view = self.rew_v
        cdef double [:] next_obs_view = self.next_obs_v
        cdef double [:] done_view = self.done_v

        self.buffer = new CppThreadSafeRingEnvironment[double,
                                                       double,
                                                       double,
                                                       double](size,
                                                               obs_dim,
                                                               act_dim,
                                                               &stored_size_view[0],
                                                               &next_index_view[0],
                                                               &obs_view[0],
                                                               &act_view[0],
                                                               &rew_view[0],
                                                               &next_obs_view[0],
                                                               &done_view[0])

        self.buffer.get_buffer_pointers(self.obs.ptr,
                                        self.act.ptr,
                                        self.rew.ptr,
                                        self.next_obs.ptr,
                                        self.done.ptr)

        self.buffer_size = self.buffer.get_buffer_size()

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def _add_N(self,
               np.ndarray[Obs     ,ndim = 2, mode="c"] obs not None,
               np.ndarray[Act     ,ndim = 2, mode="c"] act not None,
               np.ndarray[Rew     ,ndim = 1, mode="c"] rew not None,
               np.ndarray[Next_Obs,ndim = 2, mode="c"] next_obs not None,
               np.ndarray[Done    ,ndim = 1, mode="c"] done not None,
               size_t N=1):
        return self.buffer.store(&obs[0,0],&act[0,0],&rew[0],
                                 &next_obs[0,0],&done[0],N)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def _add_1(self,
               np.ndarray[Obs     ,ndim = 1, mode="c"] obs not None,
               np.ndarray[Act     ,ndim = 1, mode="c"] act not None,
               Rew rew,
               np.ndarray[Next_Obs,ndim = 1, mode="c"] next_obs not None,
               double done):
        return self.buffer.store(&obs[0],&act[0],&rew,&next_obs[0],&done,1)

    def clear(self):
        return self.buffer.clear()

    def get_stored_size(self):
        return self.buffer.get_stored_size()

    def get_next_index(self):
        return self.buffer.get_next_index()

cdef class SelectiveEnvironment(Environment):
    cdef CppSelectiveEnvironment[double,double,double,double] *buffer
    def __cinit__(self,episode_len,obs_dim,act_dim,*,Nepisodes=10,**kwargs):
        self.buffer_size = episode_len * Nepisodes

        self.buffer = new CppSelectiveEnvironment[double,double,
                                                  double,double](episode_len,
                                                                 Nepisodes,
                                                                 obs_dim,
                                                                 act_dim)

        self.buffer.get_buffer_pointers(self.obs.ptr,
                                        self.act.ptr,
                                        self.rew.ptr,
                                        self.next_obs.ptr,
                                        self.done.ptr)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def _add_N(self,
               np.ndarray[Obs     ,ndim = 2, mode="c"] obs not None,
               np.ndarray[Act     ,ndim = 2, mode="c"] act not None,
               np.ndarray[Rew     ,ndim = 1, mode="c"] rew not None,
               np.ndarray[Next_Obs,ndim = 2, mode="c"] next_obs not None,
               np.ndarray[Done    ,ndim = 1, mode="c"] done not None,
               size_t N=1):
        return self.buffer.store(&obs[0,0],&act[0,0],&rew[0],
                                 &next_obs[0,0],&done[0],N)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def _add_1(self,
               np.ndarray[Obs     ,ndim = 1, mode="c"] obs not None,
               np.ndarray[Act     ,ndim = 1, mode="c"] act not None,
               Rew rew,
               np.ndarray[Next_Obs,ndim = 1, mode="c"] next_obs not None,
               double done):
        return self.buffer.store(&obs[0],&act[0],&rew,&next_obs[0],&done,1)

    def clear(self):
        return self.buffer.clear()

    def get_stored_size(self):
        return self.buffer.get_stored_size()

    def get_next_index(self):
        return self.buffer.get_next_index()

    def get_stored_episode_size(self):
        return self.buffer.get_stored_episode_size()

    def delete_episode(self,i):
        return self.buffer.delete_episode(i)

    def get_episode(self,i):
        cdef size_t len = 0
        self.buffer.get_episode(i,len,
                                self.obs.ptr,self.act.ptr,self.rew.ptr,
                                self.next_obs.ptr,self.done.ptr)
        if len == 0:
            return {'obs': np.ndarray((0,self.obs_dim)),
                    'act': np.ndarray((0,self.act_dim)),
                    'rew': np.ndarray((0)),
                    'next_obs': np.ndarray((0,self.obs_dim)),
                    'done': np.ndarray(0)}

        self.obs.update_vec_size(len)
        self.act.update_vec_size(len)
        self.rew.update_vec_size(len)
        self.next_obs.update_vec_size(len)
        self.done.update_vec_size(len)
        return {'obs': np.asarray(self.obs),
                'act': np.asarray(self.act),
                'rew': np.asarray(self.rew),
                'next_obs': np.asarray(self.next_obs),
                'done': np.asarray(self.done)}

    def _encode_sample(self,indexes):
        self.buffer.get_buffer_pointers(self.obs.ptr,
                                        self.act.ptr,
                                        self.rew.ptr,
                                        self.next_obs.ptr,
                                        self.done.ptr)
        cdef size_t buffer_size = self.get_buffer_size()
        self.obs.update_vec_size(buffer_size)
        self.act.update_vec_size(buffer_size)
        self.rew.update_vec_size(buffer_size)
        self.next_obs.update_vec_size(buffer_size)
        self.done.update_vec_size(buffer_size)
        return super()._encode_sample(indexes)

cdef class ReplayBuffer(RingEnvironment):
    def __cinit__(self,size,obs_dim,act_dim,**kwargs):
        pass

    def sample(self,batch_size):
        cdef idx = np.random.randint(0,self.get_stored_size(),batch_size)
        return self._encode_sample(idx)

cdef class ProcessSharedReplayBuffer(ProcessSharedRingEnvironment):
    def __cinit__(self,size,obs_dim,act_dim,**kwargs):
        pass

    def sample(self,batch_size):
        cdef idx = np.random.randint(0,self.get_stored_size(),batch_size)
        return self._encode_sample(idx)

    def init_worker(self):
        return ProcessSharedRingEnvironment(self.buffer_size,
                                            self.obs_dim,self.act_dim,
                                            stored_size = self.stored_size_v,
                                            next_index = self.next_index_v,
                                            obs = self.obs_v,
                                            act = self.act_v,
                                            rew = self.rew_v,
                                            next_obs = self.next_obs_v,
                                            done = self.done_v)

cdef class SelectiveReplayBuffer(SelectiveEnvironment):
    def __cinit__(self,episode_len,obs_dim,act_dim,*,Nepisodes=10,**kwargs):
        pass

    def sample(self,batch_size):
        cdef idx = np.random.randint(0,self.get_stored_size(),batch_size)
        return self._encode_sample(idx)

cdef class PrioritizedReplayBuffer(RingEnvironment):
    cdef VectorDouble weights
    cdef VectorSize_t indexes
    cdef double alpha
    cdef CppPrioritizedSampler[double]* per
    def __cinit__(self,size,obs_dim,act_dim,*,alpha=0.6,**kwrags):
        self.alpha = alpha
        self.per = new CppPrioritizedSampler[double](size,alpha)
        self.weights = VectorDouble()
        self.indexes = VectorSize_t()

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def _update_1(self,size_t next_index):
        self.per.set_priorities(next_index)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def _update_1p(self,size_t next_index,Prio p):
        self.per.set_priorities(next_index,p)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def _update_N(self,size_t next_index,size_t N=1):
        self.per.set_priorities(next_index,N,self.get_buffer_size())

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def _update_Np(self,size_t next_index,Prio [:] p,size_t N=1):
        self.per.set_priorities(next_index,&p[0],N,self.get_buffer_size())

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def _add(self,obs,act,rew,next_obs,done):
        cdef size_t next_index
        cdef size_t N
        if obs.ndim == 1:
            next_index = self._add_1(obs,act,rew,next_obs,done)
            self._update_1(next_index)
        else:
            N = obs.shape[0]
            next_index = self._add_N(obs,act,rew,next_obs,done,N)
            self._update_N(next_index,N)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def _add_p(self,obs,act,rew,next_obs,done,priorities):
        cdef size_t next_index
        cdef size_t N
        if obs.ndim == 1:
            next_index = self._add_1(obs,act,rew,next_obs,done)
            self._update_1p(next_index,priorities)
        else:
            N = obs.shape[0]
            next_index = self._add_N(obs,act,rew,next_obs,done,N)
            self._update_Np(next_index,priorities,N)

    def add(self,obs,act,rew,next_obs,done,priorities = None):
        if priorities is not None:
            self._add_p(obs,act,rew,next_obs,done,priorities)
        else:
            self._add(obs,act,rew,next_obs,done)

    def sample(self,batch_size,beta = 0.4):
        self.per.sample(batch_size,beta,
                        self.weights.vec,self.indexes.vec,
                        self.get_stored_size())
        idx = np.asarray(self.indexes)
        samples = self._encode_sample(idx)
        samples['weights'] = np.asarray(self.weights)
        samples['indexes'] = idx
        return samples

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def _update_priorities(self,
                           np.ndarray[size_t,ndim = 1, mode="c"] indexes    not None,
                           np.ndarray[Prio  ,ndim = 1, mode="c"] priorities not None,
                           size_t N=1):
        self.per.update_priorities(&indexes[0],&priorities[0],N)

    def update_priorities(self,indexes,priorities):
        cdef idx = np.asarray(np.ravel(indexes),dtype=np.uint64)
        cdef ps = np.asarray(np.ravel(priorities),dtype=np.float64)
        cdef size_t N = idx.shape[0]
        self._update_priorities(idx,priorities,N)

    def clear(self):
        super().clear()
        self.per.clear()

    def get_max_priority(self):
        return self.per.get_max_priority()

cdef class ProcessSharedPrioritizedWorker(ProcessSharedRingEnvironment):
    cdef VectorDouble weights
    cdef VectorSize_t indexes
    cdef double alpha
    cdef CppThreadSafePrioritizedSampler[double]* per
    def __cinit__(self,size,obs_dim,act_dim,*,alpha=0.6,**kwrags):
        self.alpha = alpha
        self.per = new CppThreadSafePrioritizedSampler[double](size,alpha)
        self.weights = VectorDouble()
        self.indexes = VectorSize_t()

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def _update_1(self,size_t next_index):
        self.per.set_priorities(next_index)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def _update_1p(self,size_t next_index,Prio p):
        self.per.set_priorities(next_index,p)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def _update_N(self,size_t next_index,size_t N=1):
        self.per.set_priorities(next_index,N,self.get_buffer_size())

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def _update_Np(self,size_t next_index,Prio [:] p,size_t N=1):
        self.per.set_priorities(next_index,&p[0],N,self.get_buffer_size())

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def _add(self,obs,act,rew,next_obs,done):
        cdef size_t next_index
        cdef size_t N
        if obs.ndim == 1:
            next_index = self._add_1(obs,act,rew,next_obs,done)
            self._update_1(next_index)
        else:
            N = obs.shape[0]
            next_index = self._add_N(obs,act,rew,next_obs,done,N)
            self._update_N(next_index,N)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def _add_p(self,obs,act,rew,next_obs,done,priorities):
        cdef size_t next_index
        cdef size_t N
        if obs.ndim == 1:
            next_index = self._add_1(obs,act,rew,next_obs,done)
            self._update_1p(next_index,priorities)
        else:
            N = obs.shape[0]
            next_index = self._add_N(obs,act,rew,next_obs,done,N)
            self._update_Np(next_index,priorities,N)

    def add(self,obs,act,rew,next_obs,done,priorities = None):
        if priorities is not None:
            self._add_p(obs,act,rew,next_obs,done,priorities)
        else:
            self._add(obs,act,rew,next_obs,done)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def _update_priorities(self,
                           np.ndarray[size_t,ndim = 1, mode="c"] indexes    not None,
                           np.ndarray[Prio  ,ndim = 1, mode="c"] priorities not None,
                           size_t N=1):
        self.per.update_priorities(&indexes[0],&priorities[0],N)

    def update_priorities(self,indexes,priorities):
        cdef idx = np.asarray(np.ravel(indexes),dtype=ctypes.c_size_t)
        cdef ps = np.asarray(np.ravel(priorities),dtype=np.float64)
        cdef size_t N = idx.shape[0]
        self._update_priorities(idx,priorities,N)

    def clear(self):
        super().clear()
        self.per.clear()

    def get_max_priority(self):
        return self.per.get_max_priority()

cdef class ProcessSharedPrioritizedReplayBuffer(ProcessSharedPrioritizedWorker):
    def sample(self,batch_size,beta = 0.4):
        self.per.sample(batch_size,beta,
                        self.weights.vec,self.indexes.vec,
                        self.get_stored_size())
        idx = np.asarray(self.indexes)
        samples = self._encode_sample(idx)
        samples['weights'] = np.asarray(self.weights)
        samples['indexes'] = idx
        return samples

    def init_worker(self):
        return ProcessSharedPrioritizedWorker(self.buffer_size,
                                              self.obs_dim,self.act_dim,
                                              alpha = self.alpha,
                                              stored_size = self.stored_size_v,
                                              next_index = self.next_index_v,
                                              obs = self.obs_v,
                                              act = self.act_v,
                                              rew = self.rew_v,
                                              next_obs = self.next_obs_v,
                                              done = self.done_v)


cdef class NstepReplayBuffer(ReplayBuffer):
    cdef CppNstepRewardBuffer[double,double]* nrb
    cdef PointerDouble gamma
    cdef PointerDouble nstep_rew
    cdef PointerDouble nstep_next_obs
    def __cinit__(self,size,obs_dim,act_dim,*,n_step = 4, discount = 0.99,**kwargs):
        self.nrb = new CppNstepRewardBuffer[double,double](size,obs_dim,
                                                           n_step,discount)
        self.gamma = PointerDouble(ndim=2,value_dim=1,size=size)
        self.nstep_rew = PointerDouble(ndim=2,value_dim=1,size=size)
        self.nstep_next_obs = PointerDouble(ndim=2,value_dim=obs_dim,size=size)

    def _encode_sample(self,indexes):
        samples = super()._encode_sample(indexes)
        cdef size_t batch_size = indexes.shape[0]

        self.nrb.sample(indexes,self.rew.ptr,self.next_obs.ptr,self.done.ptr)
        self.nrb.get_buffer_pointers(self.gamma.ptr,
                                     self.nstep_rew.ptr,
                                     self.nstep_next_obs.ptr)
        self.gamma.update_vec_size(batch_size)
        self.nstep_rew.update_vec_size(batch_size)
        self.nstep_next_obs.update_vec_size(batch_size)
        samples['discounts'] = np.asarray(self.gamma)
        samples['rew'] = np.asarray(self.nstep_rew)
        samples['next_obs'] = np.asarray(self.nstep_next_obs)
        return samples

cdef class NstepPrioritizedReplayBuffer(PrioritizedReplayBuffer):
    cdef CppNstepRewardBuffer[double,double]* nrb
    cdef PointerDouble gamma
    cdef PointerDouble nstep_rew
    cdef PointerDouble nstep_next_obs
    def __cinit__(self,size,obs_dim,act_dim,*,
                  alpha = 0.6,n_step = 4, discount = 0.99,**kwargs):
        self.nrb = new CppNstepRewardBuffer[double,double](size,obs_dim,
                                                           n_step,discount)
        self.gamma = PointerDouble(ndim=2,value_dim=1,size=size)
        self.nstep_rew = PointerDouble(ndim=2,value_dim=1,size=size)
        self.nstep_next_obs = PointerDouble(ndim=2,value_dim=obs_dim,size=size)

    def _encode_sample(self,indexes):
        samples = super()._encode_sample(indexes)
        cdef size_t batch_size = indexes.shape[0]

        self.nrb.sample(indexes,self.rew.ptr,self.next_obs.ptr,self.done.ptr)
        self.nrb.get_buffer_pointers(self.gamma.ptr,
                                     self.nstep_rew.ptr,
                                     self.nstep_next_obs.ptr)
        self.gamma.update_vec_size(batch_size)
        self.nstep_rew.update_vec_size(batch_size)
        self.nstep_next_obs.update_vec_size(batch_size)
        samples['discounts'] = np.asarray(self.gamma)
        samples['rew'] = np.asarray(self.nstep_rew)
        samples['next_obs'] = np.asarray(self.nstep_next_obs)
        return samples
