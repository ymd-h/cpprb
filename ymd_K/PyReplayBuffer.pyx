# distutils: language = c++

from libc.stdlib cimport malloc, free
from cython.operator cimport dereference
cimport numpy as np
import numpy as np
import cython

from ymd_K cimport ReplayBuffer

cdef class VectorWrapper:
    cdef Py_ssize_t *shape
    cdef Py_ssize_t *strides
    cdef Py_ssize_t itemsize
    cdef int ndim
    cdef int value_dim

    def __cinit__(self):
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

    def __cinit__(self,value_dim=1):
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

    def __cinit__(self,value_dim=1):
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

    def __cinit__(self,value_dim=1):
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

    def __cinit__(self,ndim,value_dim,size):
        self.itemsize = sizeof(double)

        self.ndim = ndim
        self.value_dim = value_dim
        self._vec_size = value_dim * size

    def vec_size(self):
        return self._vec_size

    cdef void set_buffer(self,Py_buffer* buffer):
        buffer.buf = <void*> self.ptr
        buffer.format = 'd'

cdef class PyReplayBuffer:
    cdef ReplayBuffer[double,double,double,double] *thisptr
    cdef PointerDouble obs
    cdef PointerDouble act
    cdef PointerDouble rew
    cdef PointerDouble next_obs
    cdef PointerDouble done
    cdef VectorULong indexes
    cdef int buffer_size
    cdef int obs_dim
    cdef int act_dim
    def __cinit__(self,size,obs_dim,act_dim):
        print("Replay Buffer")
        self.buffer_size = size
        self.obs_dim = obs_dim
        self.act_dim = act_dim

        self.thisptr = new ReplayBuffer[double,double,double,double](size,
                                                                     obs_dim,
                                                                     act_dim)
        self.obs = PointerDouble(2,obs_dim,size)
        self.act = PointerDouble(2,act_dim,size)
        self.rew = PointerDouble(1,1,size)
        self.next_obs = PointerDouble(2,obs_dim,size)
        self.done = PointerDouble(1,1,size)
        self.indexes = VectorULong()

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def _add_N(self,
               np.ndarray[double, ndim=2, mode="c"] obs not None,
               np.ndarray[double, ndim=2, mode="c"] act not None,
               np.ndarray[double, ndim=1, mode="c"] rew not None,
               np.ndarray[double, ndim=2, mode="c"] next_obs not None,
               np.ndarray[double, ndim=1, mode="c"] done not None,
               size_t N=1):
        self.thisptr.add(&obs[0,0],&act[0,0],&rew[0],&next_obs[0,0],&done[0],N)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def _add_1(self,
               np.ndarray[double, ndim=1, mode="c"] obs not None,
               np.ndarray[double, ndim=1, mode="c"] act not None,
               double rew,
               np.ndarray[double, ndim=1, mode="c"] next_obs not None,
               double done):
        self.thisptr.add(&obs[0],&act[0],&rew,&next_obs[0],&done,1)

    def add(self,obs,act,rew,next_obs,done):
        if obs.ndim == 1:
            self._add_1(obs,act,rew,next_obs,done)
        else:
            self._add_N(obs,act,rew,next_obs,done,obs.shape[0])

    def sample(self,size):
        self.thisptr.sample(size,
                            self.obs.ptr,
                            self.act.ptr,
                            self.rew.ptr,
                            self.next_obs.ptr,
                            self.done.ptr,
                            self.indexes.vec)
        idx = np.asarray(self.indexes)
        return {'obs': np.asarray(self.obs)[idx,:],
                'act': np.asarray(self.act)[idx,:],
                'rew': np.asarray(self.rew)[idx],
                'next_obs': np.asarray(self.next_obs)[idx,:],
                'done': np.asarray(self.done)[idx]}

    def get_buffer_size(self):
        return self.buffer_size

cdef class PyPrioritizedReplayBuffer:
    cdef PrioritizedReplayBuffer[double,double,double,double,double] *thisptr
    cdef PointerDouble obs
    cdef PointerDouble act
    cdef PointerDouble rew
    cdef PointerDouble next_obs
    cdef PointerDouble done
    cdef VectorDouble weights
    cdef VectorULong indexes
    cdef int buffer_size
    cdef int obs_dim
    cdef int act_dim
    cdef double alpha
    def __cinit__(self,size,obs_dim,act_dim,alpha=0.6):
        print("Prioritized Replay Buffer")
        self.buffer_size = size
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.alpha = alpha

        self.thisptr = new PrioritizedReplayBuffer[double,double,
                                                   double,double,double](size,
                                                                         obs_dim,
                                                                         act_dim,
                                                                         alpha)
        self.obs = PointerDouble(2,obs_dim,size)
        self.act = PointerDouble(2,act_dim,size)
        self.rew = PointerDouble(1,1,size)
        self.next_obs = PointerDouble(2,obs_dim,size)
        self.done = PointerDouble(1,1,size)
        self.weights = VectorDouble()
        self.indexes = VectorULong()

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def _add_N(self,
               np.ndarray[double, ndim=2, mode="c"] obs not None,
               np.ndarray[double, ndim=2, mode="c"] act not None,
               np.ndarray[double, ndim=1, mode="c"] rew not None,
               np.ndarray[double, ndim=2, mode="c"] next_obs not None,
               np.ndarray[double, ndim=1, mode="c"] done not None,
               size_t N=1):
        self.thisptr.add(&obs[0,0],&act[0,0],&rew[0],&next_obs[0,0],&done[0],N)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def _add_1(self,
               np.ndarray[double, ndim=1, mode="c"] obs not None,
               np.ndarray[double, ndim=1, mode="c"] act not None,
               double rew,
               np.ndarray[double, ndim=1, mode="c"] next_obs not None,
               double done):
        self.thisptr.add(&obs[0],&act[0],&rew,&next_obs[0],&done)

    def add(self,obs,act,rew,next_obs,done):
        if obs.ndim == 1:
            self._add_1(obs,act,rew,next_obs,done)
        else:
            self._add_N(obs,act,rew,next_obs,done,obs.shape[0])

    def sample(self,size,beta):
        self.thisptr.sample(size,beta,
                            self.obs.ptr,
                            self.act.ptr,
                            self.rew.ptr,
                            self.next_obs.ptr,
                            self.done.ptr,
                            self.weights.vec,
                            self.indexes.vec)
        idx = np.asarray(self.indexes)
        return {'obs': np.asarray(self.obs)[idx,:],
                'act': np.asarray(self.act)[idx,:],
                'rew': np.asarray(self.rew)[idx],
                'next_obs': np.asarray(self.next_obs)[idx,:],
                'done': np.asarray(self.done)[idx],
                'weights': np.asarray(self.weights),
                'indexes': idx}

    def update_priorities(self,indexes,priorities):
        self.thisptr.update_priorities(indexes,priorities)

    def get_buffer_size(self):
        return self.buffer_size
