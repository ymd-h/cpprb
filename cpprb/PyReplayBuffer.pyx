# distutils: language = c++

from libc.stdlib cimport malloc, free
from cython.operator cimport dereference
cimport numpy as np
import numpy as np
import cython

from cpprb cimport ReplayBuffer

cdef class VectorWrapper:
    cdef Py_ssize_t *shape
    cdef Py_ssize_t *strides
    cdef Py_ssize_t itemsize
    cdef int ndim
    cdef int value_dim

    def __cinit__(self,**kwarg):
        self.shape   = <Py_ssize_t*>malloc(sizeof(Py_ssize_t) * 2)
        self.strides = <Py_ssize_t*>malloc(sizeof(Py_ssize_t) * 2)

    def __dealloc__(self):
        free(self.shape)
        free(self.strides)

    cdef void update_size(self):
        self.shape[0] = <Py_ssize_t>(self.vec_size()//self.value_dim)
        self.strides[self.ndim -1] = <Py_ssize_t> self.itemsize

        if self.ndim is 2:
            self.shape[1] = <Py_ssize_t> (self.value_dim)
            self.strides[0] = self.value_dim * <Py_ssize_t> self.itemsize

    cdef void set_buffer(self,Py_buffer *buffer):
        pass

    def __getbuffer__(self, Py_buffer *buffer, int flags):
        # relevant documentation http://cython.readthedocs.io/en/latest/src/userguide/buffer.html#a-matrix-class

        self.update_size()

        self.set_buffer(buffer)
        buffer.len = self.vec_size() * self.itemsize
        buffer.readonly = 0
        buffer.ndim = self.ndim
        buffer.shape = self.shape
        buffer.strides = self.strides
        buffer.suboffsets = NULL
        buffer.itemsize = self.itemsize
        buffer.internal = NULL
        buffer.obj = self

    def __releasebuffer__(self, Py_buffer *buffer):
        pass

cdef class VectorInt(VectorWrapper):
    cdef vector[int] vec

    def __cinit__(self,*,value_dim=1,**kwargs):
        self.vec = vector[int]()
        self.itemsize = sizeof(int)

        self.ndim = 1 if value_dim is 1 else 2
        self.value_dim = value_dim

    def vec_size(self):
        return self.vec.size()

    cdef void set_buffer(self,Py_buffer* buffer):
        buffer.buf = <void*>(self.vec.data())
        buffer.format = 'i'

cdef class VectorDouble(VectorWrapper):
    cdef vector[double] vec

    def __cinit__(self,*,value_dim=1,**kwargs):
        self.vec = vector[double]()
        self.itemsize = sizeof(double)

        self.ndim = 1 if value_dim is 1 else 2
        self.value_dim = value_dim

    def vec_size(self):
        return self.vec.size()

    cdef void set_buffer(self,Py_buffer* buffer):
        buffer.buf = <void*>(self.vec.data())
        buffer.format = 'd'

cdef class VectorULong(VectorWrapper):
    cdef vector[size_t] vec

    def __cinit__(self,*,value_dim=1,**kwargs):
        self.vec = vector[size_t]()
        self.itemsize = sizeof(size_t)

        self.ndim = 1 if value_dim is 1 else 2
        self.value_dim = value_dim

    def vec_size(self):
        return self.vec.size()

    cdef void set_buffer(self,Py_buffer* buffer):
        buffer.buf = <void*>(self.vec.data())
        buffer.format = 'L'

cdef class PointerDouble(VectorWrapper):
    cdef double* ptr
    cdef int _vec_size

    def __cinit__(self,*,ndim=1,value_dim=1,size=1,**kwargs):
        self.itemsize = sizeof(double)

        self.ndim = ndim
        self.value_dim = value_dim
        self._vec_size = value_dim * size

    def vec_size(self):
        return self._vec_size

    cdef void set_buffer(self,Py_buffer* buffer):
        buffer.buf = <void*> self.ptr
        buffer.format = 'd'

    cdef void update_vec_size(self,size):
        self._vec_size = self.value_dim * size

cdef class PyInternalBuffer:
    cdef InternalBuffer[double,double,double,double] *buffer
    cdef PointerDouble obs
    cdef PointerDouble act
    cdef PointerDouble rew
    cdef PointerDouble next_obs
    cdef PointerDouble done
    cdef int buffer_size
    cdef int obs_dim
    cdef int act_dim
    def __cinit__(self,size,obs_dim,act_dim,**kwargs):
        self.buffer_size = size
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.obs = PointerDouble(ndim=2,value_dim=obs_dim,size=size)
        self.act = PointerDouble(ndim=2,value_dim=act_dim,size=size)
        self.rew = PointerDouble(ndim=1,value_dim=1,size=size)
        self.next_obs = PointerDouble(ndim=2,value_dim=obs_dim,size=size)
        self.done = PointerDouble(ndim=1,value_dim=1,size=size)

        self.buffer = new InternalBuffer[double,double,double,double](size,
                                                                      obs_dim,
                                                                      act_dim)

        self.buffer.get_buffer_pointers(self.obs.ptr,
                                        self.act.ptr,
                                        self.rew.ptr,
                                        self.next_obs.ptr,
                                        self.done.ptr);
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def _add_N(self,
               np.ndarray[double, ndim=2, mode="c"] obs not None,
               np.ndarray[double, ndim=2, mode="c"] act not None,
               np.ndarray[double, ndim=1, mode="c"] rew not None,
               np.ndarray[double, ndim=2, mode="c"] next_obs not None,
               np.ndarray[double, ndim=1, mode="c"] done not None,
               size_t N=1):
        self.buffer.store(&obs[0,0],&act[0,0],&rew[0],&next_obs[0,0],&done[0],N)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def _add_1(self,
               np.ndarray[double, ndim=1, mode="c"] obs not None,
               np.ndarray[double, ndim=1, mode="c"] act not None,
               double rew,
               np.ndarray[double, ndim=1, mode="c"] next_obs not None,
               double done):
        self.buffer.store(&obs[0],&act[0],&rew,&next_obs[0],&done,1)

    def add(self,obs,act,rew,next_obs,done):
        if obs.ndim == 1:
            self._add_1(obs,act,rew,next_obs,done)
        else:
            self._add_N(obs,act,rew,next_obs,done,obs.shape[0])

    def _encode_sample(self,idx):
        return {'obs': np.asarray(self.obs)[idx,:],
                'act': np.asarray(self.act)[idx,:],
                'rew': np.asarray(self.rew)[idx],
                'next_obs': np.asarray(self.next_obs)[idx,:],
                'done': np.asarray(self.done)[idx]}

    def get_buffer_size(self):
        return self.buffer_size

    def clear(self):
        return self.buffer.clear()

    def get_stored_size(self):
        return self.buffer.get_stored_size()

    def get_next_index(self):
        return self.buffer.get_next_index()

cdef class PyReplayBuffer(PyInternalBuffer):
    def __cinit__(self,size,obs_dim,act_dim,**kwargs):
        print("Replay Buffer")

    def sample(self,batch_size):
        idx = np.random.randint(0,self.get_stored_size(),batch_size)
        return self._encode_sample(idx)

cdef class PyPrioritizedReplayBuffer(PyInternalBuffer):
    cdef VectorDouble weights
    cdef VectorULong indexes
    cdef double alpha
    cdef PrioritizedSampler[double]* per
    def __cinit__(self,size,obs_dim,act_dim,*,alpha=0.6,**kwrags):
        print("Prioritized Replay Buffer")
        self.alpha = alpha

        self.per = new PrioritizedSampler[double](size,alpha)
        self.weights = VectorDouble()
        self.indexes = VectorULong()

    cdef _update_1(self,size_t next_index):
        self.per.set_priorities(next_index)

    cdef _update_1p(self,size_t next_index,double p):
        self.per.set_priorities(next_index,p)

    cdef _update_N(self,size_t next_index,size_t N=1):
        self.per.set_priorities(next_index,N,self.get_stored_size())

    cdef _update_Np(self,size_t next_index,np.ndarray[double,ndim=1] p,size_t N=1):
        self.per.set_priorities(next_index,&p[0],N,self.get_stored_size())

    def add(self,obs,act,rew,next_obs,done,priorities = None):
        cdef size_t next_index = self.get_next_index()
        if obs.ndim == 1:
            self._add_1(obs,act,rew,next_obs,done)
            if priorities:
                self._update_1p(next_index,priorities)
            else:
                self._update_1(next_index)
        else:
            N = obs.shape[0]
            self._add_N(obs,act,rew,next_obs,done,N)
            if priorities:
                self._update_Np(next_index,priorities,N)
            else:
                self._update_N(next_index,N)

    def sample(self,batch_size,beta):
        self.per.sample(batch_size,beta,
                        self.weights.vec,self.indexes.vec,
                        self.get_stored_size())
        idx = np.asarray(self.indexes)
        samples = self._encode_sample(idx)
        samples['weights'] = np.asarray(self.weights)
        samples['indexes'] = idx
        return samples

    def update_priorities(self,indexes,priorities):
        self.per.update_priorities(indexes,priorities)

    def clear(self):
        super().clear()
        self.per.clear()

    def get_max_priority(self):
        return self.per.get_max_priority()

cdef class PyNstepReplayBuffer(PyReplayBuffer):
    cdef NstepRewardBuffer[double,double]* nrb
    cdef PointerDouble gamma
    cdef PointerDouble nstep_rew
    cdef PointerDouble nstep_next_obs
    def __cinit__(self,size,obs_dim,act_dim,*,n_step = 4, discount = 0.99,**kwargs):
        self.nrb = new NstepRewardBuffer[double,double](size,obs_dim,n_step,discount)
        self.gamma = PointerDouble(ndim=1,value_dim=1,size=size)
        self.nstep_rew = PointerDouble(ndim=1,value_dim=1,size=size)
        self.nstep_next_obs = PointerDouble(ndim=2,value_dim=obs_dim,size=size)

        self.nrb.get_buffer_pointers(self.gamma.ptr,
                                     self.nstep_rew.ptr,
                                     self.nstep_next_obs.ptr)

    def _encode_sample(self,indexes):
        self.nrb.sample(indexes,self.rew.ptr,self.next_obs.ptr,self.done.ptr)
        samples = super()._encode_sample(indexes)
        samples['discounts'] = np.asarray(self.gamma)[indexes]
        samples['rew'] = np.asarray(self.nstep_rew)[indexes]
        samples['next_obs'] = np.asarray(self.nstep_next_obs)[indexes]
        return samples

cdef class PyNstepPrioritizedReplayBuffer(PyPrioritizedReplayBuffer):
    cdef NstepRewardBuffer[double,double]* nrb
    cdef PointerDouble gamma
    cdef PointerDouble nstep_rew
    cdef PointerDouble nstep_next_obs
    def __cinit__(self,size,obs_dim,act_dim,*,
                  alpha = 0.6,n_step = 4, discount = 0.99,**kwargs):
        self.nrb = new NstepRewardBuffer[double,double](size,obs_dim,n_step,discount)
        self.gamma = PointerDouble(ndim=1,value_dim=1,size=size)
        self.nstep_rew = PointerDouble(ndim=1,value_dim=1,size=size)
        self.nstep_next_obs = PointerDouble(ndim=2,value_dim=obs_dim,size=size)

    def _encode_sample(self,indexes):
        self.nrb.sample(indexes,self.rew.ptr,self.next_obs.ptr,self.done.ptr)
        samples = super()._encode_sample(indexes)

        self.nrb.get_buffer_pointers(self.gamma.ptr,
                                     self.nstep_rew.ptr,
                                     self.nstep_next_obs.ptr)

        batch_size = indexes.shap[0]
        samples['discounts'] = np.asarray(self.gamma)[:batch_size]
        samples['rew'] = np.asarray(self.nstep_rew)[:batch_size]
        samples['next_obs'] = np.asarray(self.nstep_next_obs)[indexes]
        return samples
