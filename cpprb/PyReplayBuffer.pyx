# distutils: language = c++

from multiprocessing.sharedctypes import RawArray
import ctypes
cimport numpy as np
import numpy as np
import cython

from cpprb.ReplayBuffer cimport *

from .VectorWrapper cimport *
from .VectorWrapper import (VectorWrapper,
                            VectorInt,VectorSize_t,VectorDouble,PointerDouble)

cdef double [::1] Cview(array):
    return np.ravel(np.array(array,copy=False,dtype=np.double,ndmin=1,order='C'))

cdef size_t [::1] Csize(array):
    return np.ravel(np.array(array,copy=False,dtype=np.uint64,ndmin=1,order='C'))

@cython.embedsignature(True)
cdef class Environment:
    """
    Base class to store environment
    """
    cdef PointerDouble obs
    cdef PointerDouble act
    cdef PointerDouble rew
    cdef PointerDouble next_obs
    cdef PointerDouble done
    cdef size_t buffer_size
    cdef size_t obs_dim
    cdef size_t act_dim
    cdef size_t rew_dim
    cdef bool is_discrete_action
    cdef obs_shape

    def __cinit__(self,size,obs_dim=1,act_dim=1,*,
                  rew_dim=1,is_discrete_action = False,
                  obs_shape = None, **kwargs):
        self.obs_shape = obs_shape
        self.is_discrete_action = is_discrete_action

        cdef size_t _dim
        if self.obs_shape is None:
            self.obs_dim = obs_dim
        else:
            self.obs_dim = 1
            for _dim in self.obs_shape:
                self.obs_dim *= _dim

        self.act_dim = act_dim if not self.is_discrete_action else 1
        self.rew_dim = rew_dim

        self.obs = PointerDouble(ndim=2,value_dim=self.obs_dim,size=size)
        self.act = PointerDouble(ndim=2,value_dim=self.act_dim,size=size)
        self.rew = PointerDouble(ndim=2,value_dim=self.rew_dim,size=size)
        self.next_obs = PointerDouble(ndim=2,value_dim=self.obs_dim,size=size)
        self.done = PointerDouble(ndim=2,value_dim=1,size=size)

    def __init__(self,size,obs_dim=1,act_dim=1,*,
                 rew_dim=1,is_discrete_action = False,
                 obs_shape = None, **kwargs):
        """
        Parameters
        ----------
        size : int
            buffer size
        obs_dim : int
            observation (obs) dimension
        act_dim : int
            action (act) dimension
        rew_dim : int, optional
            reward (rew) dimension whose default value is 1
        is_discrete_action: bool, optional
            If True, act_dim is compressed to 1 whose default value is False
        """
        pass

    cdef size_t _add(self,double [::1] o,double [::1] a,double [::1] r,
                     double [::1] no,double [::1] d):
        raise NotImplementedError

    def add(self,obs,act,rew,next_obs,done):
        """
        Add environment(s) into replay buffer.
        Multiple step environments can be added.

        Parameters
        ----------
        obs : array_like or float or int
            observation(s)
        act : array_like or float or int
            action(s)
        rew : array_like or float or int
            reward(s)
        next_obs : array_like or float or int
            next observation(s)
        done : array_like or float or int
            done(s)

        Returns
        -------
        int
            the stored first index
        """
        return self._add(Cview(obs),Cview(act),Cview(rew),Cview(next_obs),Cview(done))

    def _encode_sample(self,idx):
        dtype = np.int if self.is_discrete_action else np.double

        _o = self.obs.as_numpy()
        if self.obs_shape:
            _o = _o.reshape((-1,*self.obs_shape))

        return {'obs': _o[idx],
                'act': self.act.as_numpy(dtype=dtype)[idx],
                'rew': self.rew.as_numpy()[idx],
                'next_obs': self.next_obs.as_numpy()[idx],
                'done': self.done.as_numpy()[idx]}

    cpdef size_t get_buffer_size(self):
        """
        Get buffer size

        Parameters
        ----------

        Returns
        -------
        size_t
            buffer size
        """
        return self.buffer_size

    cdef void _update_size(self,size_t new_size):
        """ Update environment size

        Parameters
        ----------
        new_size : size_t
            new size to set as environment (obs,act,rew,next_obs,done)

        Returns
        -------
        """
        self.obs.update_vec_size(new_size)
        self.act.update_vec_size(new_size)
        self.rew.update_vec_size(new_size)
        self.next_obs.update_vec_size(new_size)
        self.done.update_vec_size(new_size)

    cpdef size_t get_obs_dim(self):
        """Return observation dimension (obs_dim)
        """
        return self.obs_dim

    cpdef size_t get_act_dim(self):
        """Return action dimension (act_dim)
        """
        return self.act_dim

    cpdef size_t get_rew_dim(self):
        """Return reward dimension (rew_dim)
        """
        return self.rew_dim

@cython.embedsignature(True)
cdef class RingEnvironment(Environment):
    """
    Ring buffer class to store environment
    """
    cdef CppRingEnvironment[double,double,double,double] *buffer
    def __cinit__(self,size,obs_dim=1,act_dim=1,*,rew_dim = 1,**kwargs):
        self.buffer = new CppRingEnvironment[double,double,double,double](size,
                                                                          obs_dim,
                                                                          act_dim,
                                                                          rew_dim)

        self.buffer.get_buffer_pointers(self.obs.ptr,
                                        self.act.ptr,
                                        self.rew.ptr,
                                        self.next_obs.ptr,
                                        self.done.ptr)

        self.buffer_size = get_buffer_size(self.buffer)
        self._update_size(self.buffer_size)

    def __init__(self,size,obs_dim=1,act_dim=1,*,rew_dim = 1,**kwargs):
        """
        Parameters
        ----------
        size : int
            buffer size
        obs_dim : int
            observation (obs, next_obs) dimension
        act_dim : int
            action (act) dimension
        rew_dim : int, optional
            reward (rew) dimension whose default value is 1
        """
        pass

    cdef size_t _add(self,double [::1] obs, double [::1] act, double [::1] rew,
                     double [::1] next_obs, double [::1] done):
        return self.buffer.store(&obs[0],&act[0],&rew[0],
                                 &next_obs[0],&done[0],done.shape[0])

    cpdef void clear(self) except *:
        """
        Clear replay buffer.

        Parameters
        ----------

        Returns
        -------
        """
        clear(self.buffer)

    cpdef size_t get_stored_size(self):
        """
        Get stored size

        Parameters
        ----------

        Returns
        -------
        size_t
            stored size
        """
        return get_stored_size(self.buffer)

    cpdef size_t get_next_index(self):
        """
        Get the next index to store

        Parameters
        ----------

        Returns
        -------
        size_t
            the next index to store
        """
        return get_next_index(self.buffer)

@cython.embedsignature(True)
cdef class ProcessSharedRingEnvironment(Environment):
    """
    Ring buffer class for environment.
    This class can be added from multiprocessing without explicit lock.
    """
    cdef CppThreadSafeRingEnvironment[double,double,double,double] *buffer
    cdef stored_size_v
    cdef next_index_v
    cdef obs_v
    cdef act_v
    cdef rew_v
    cdef next_obs_v
    cdef done_v
    def __cinit__(self,size,obs_dim=1,act_dim=1,*,
                  rew_dim = 1,
                  stored_size=None,next_index=None,
                  obs=None,act=None,rew=None,next_obs=None,done=None,
                  **kwargs):

        cdef size_t bsize = size
        cdef size_t N = 1
        while N < bsize:
            N *= 2

        self.stored_size_v = stored_size or RawArray(ctypes.c_size_t,1)
        self.next_index_v  = next_index  or RawArray(ctypes.c_size_t,1)
        self.obs_v         = obs         or RawArray(ctypes.c_double,N*obs_dim)
        self.act_v         = act         or RawArray(ctypes.c_double,N*act_dim)
        self.rew_v         = rew         or RawArray(ctypes.c_double,N*rew_dim)
        self.next_obs_v    = next_obs    or RawArray(ctypes.c_double,N*obs_dim)
        self.done_v        = done        or RawArray(ctypes.c_double,N)

        cdef size_t [:] stored_size_view = self.stored_size_v
        cdef size_t [:] next_index_view  = self.next_index_v
        cdef double [:] obs_view         = self.obs_v
        cdef double [:] act_view         = self.act_v
        cdef double [:] rew_view         = self.rew_v
        cdef double [:] next_obs_view    = self.next_obs_v
        cdef double [:] done_view        = self.done_v

        self.buffer = new CppThreadSafeRingEnvironment[double,
                                                       double,
                                                       double,
                                                       double](N,
                                                               obs_dim,
                                                               act_dim,
                                                               rew_dim,
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

        self.buffer_size = get_buffer_size(self.buffer)
        self._update_size(self.buffer_size)
        if N != self.buffer_size:
            raise ValueError("Size mismutch")

    def __init__(self,size,obs_dim=1,act_dim=1,*,
                 rew_dim = 1,
                 stored_size=None,next_index=None,
                 obs=None,act=None,rew=None,next_obs=None,done=None,
                 **kwargs):
        """
        Parameters
        ----------
        size : int
            buffer size
        obs_dim : int
            observation (obs, next_obs) dimension
        act_dim : int
            action (act) dimension
        rew_dim : int, optional
            reward (rew) dimension
        stored_size : multiprocessing.RawArray of ctypes.c_size_t, optional
            shared memory for stored_size. If None, new memory is allocated.
        next_index : multiprocessing.RawArray of ctypes.c_size_t, optional
            shared memory for next_index. If None, new memory is allocated.
        obs : multiprocessing.RawArray of ctypes.c_double
            shared memory for obs. If None, new memory is allocated.
        act : multiprocessing.RawArray of ctypes.c_double
            shared memory for act. If None, new memory is allocated.
        rew : multiprocessing.RawArray of ctypes.c_double
            shared memory for rew. If None, new memory is allocated.
        next_obs : multiprocessing.RawArray of ctypes.c_double
            shared memory for next_obs. If None, new memory is allocated.
        done : multiprocessing.RawArray of ctypes.c_double
            shared memory for done. If None, new memory is allocated.

        Notes
        -----
        Shared memory arguments are designed to use in init_worker method.
        Usually, you don't need to specify manually.
        """
        pass

    cdef size_t _add(self,double [::1] obs,double [::1] act, double [::1] rew,
              double [::1] next_obs, double [::1] done):
        return self.buffer.store(&obs[0],&act[0],&rew[0],
                                 &next_obs[0],&done[0],done.shape[0])

    cpdef void clear(self) except *:
        """
        Clear replay buffer.

        Parameters
        ----------

        Returns
        -------
        """
        clear(self.buffer)

    cpdef size_t get_stored_size(self):
        """
        Get stored size

        Parameters
        ----------

        Returns
        -------
        size_t
            stored size
        """
        return get_stored_size(self.buffer)

    cpdef size_t get_next_index(self):
        """
        Get the next index to store

        Parameters
        ----------

        Returns
        -------
        size_t
            the next index to store
        """
        return get_next_index(self.buffer)

@cython.embedsignature(True)
cdef class SelectiveEnvironment(Environment):
    """
    Base class for episode level management envirionment
    """
    cdef CppSelectiveEnvironment[double,double,double,double] *buffer
    def __cinit__(self,episode_len,obs_dim=1,act_dim=1,*,Nepisodes=10,rew_dim=1,**kwargs):
        self.buffer_size = episode_len * Nepisodes

        self.buffer = new CppSelectiveEnvironment[double,double,
                                                  double,double](episode_len,
                                                                 Nepisodes,
                                                                 obs_dim,
                                                                 act_dim,
                                                                 rew_dim)

        self.buffer.get_buffer_pointers(self.obs.ptr,
                                        self.act.ptr,
                                        self.rew.ptr,
                                        self.next_obs.ptr,
                                        self.done.ptr)

    def __init__(self,episode_len,obs_dim=1,act_dim=1,*,Nepisodes=10,rew_dim=1,**kwargs):
        """
        Parameters
        ----------
        episode_len : int
            the mex size of environments in a single episode
        obs_dim : int
            observation (obs, next_obs) dimension
        act_dim : int
            action (act) dimension
        Nepisodes : int, optional
            the max size of stored episodes
        rew_dim : int, optional
            reward (rew) dimension
        """
        pass

    cdef size_t _add(self,double [::1] obs,double [::1] act, double [::1] rew,
                     double [::1] next_obs, double [::1] done):
        return self.buffer.store(&obs[0],&act[0],&rew[0],
                                 &next_obs[0],&done[0],done.shape[0])

    cpdef void clear(self) except *:
        """
        Clear replay buffer.

        Parameters
        ----------

        Returns
        -------
        """
        clear(self.buffer)

    cpdef size_t get_stored_size(self):
        """
        Get stored size

        Parameters
        ----------

        Returns
        -------
        size_t
            stored size
        """
        return get_stored_size(self.buffer)

    cpdef size_t get_next_index(self):
        """
        Get the next index to store

        Parameters
        ----------

        Returns
        -------
        size_t
            the next index to store
        """
        return get_next_index(self.buffer)

    cpdef size_t get_stored_episode_size(self):
        """
        Get the size of stored episodes

        Parameters
        ----------

        Returns
        -------
        size_t
            the size of stored episodes
        """
        return self.buffer.get_stored_episode_size()

    cpdef size_t delete_episode(self,i):
        """
        Delete specified episode

        The stored environment after specified episode are moved to backward.

        Parameters
        ----------
        i : int
            the index of delete episode

        Returns
        -------
        size_t
            the size of environments in the deleted episodes
        """
        return self.buffer.delete_episode(i)

    def get_episode(self,i):
        """
        Get specified episode

        Parameters
        ----------
        i : int
            the index of extracted episode

        Returns
        -------
        dict of ndarray
            the set environment in i-th episode
        """
        cdef size_t len = 0
        self.buffer.get_episode(i,len,
                                self.obs.ptr,self.act.ptr,self.rew.ptr,
                                self.next_obs.ptr,self.done.ptr)
        if len == 0:
            return {'obs': np.ndarray((0,self.obs_dim)),
                    'act': np.ndarray((0,self.act_dim)),
                    'rew': np.ndarray((0,self.rew_dim)),
                    'next_obs': np.ndarray((0,self.obs_dim)),
                    'done': np.ndarray(0)}

        self._update_size(len)
        return {'obs': self.obs.as_numpy(),
                'act': self.act.as_numpy(),
                'rew': self.rew.as_numpy(),
                'next_obs': self.next_obs.as_numpy(),
                'done': self.done.as_numpy()}

    def _encode_sample(self,indexes):
        self.buffer.get_buffer_pointers(self.obs.ptr,
                                        self.act.ptr,
                                        self.rew.ptr,
                                        self.next_obs.ptr,
                                        self.done.ptr)
        cdef size_t buffer_size = self.get_buffer_size()
        self._update_size(buffer_size)
        return super()._encode_sample(indexes)

@cython.embedsignature(True)
cdef class ReplayBuffer(RingEnvironment):
    """
    Replay Buffer class to store environments and to sample them randomly.

    The envitonment contains observation (obs), action (act), reward (rew),
    the next observation (next_obs), and done (done).

    In this class, sampling is random sampling and the same environment can be
    chosen multiple times.
    """
    def __cinit__(self,size,obs_dim=1,act_dim=1,*,rew_dim = 1,**kwargs):
        pass

    def __init__(self,size,obs_dim=1,act_dim=1,*,rew_dim = 1,**kwargs):
        """
        Parameters
        ----------
        size : int
            buffer size
        obs_dim : int
            observation (obs and next_obs) dimension
        act_dim : int
            action (act) dimension
        rew_dim : int, optional, keyword only
            reward (rew) dimension, whose default value is 1
        """
        pass

    def sample(self,batch_size):
        """
        Sample the stored environment randomly with speciped size

        Parameters
        ----------
        batch_size : int
            sampled batch size

        Returns
        -------
        sample : dict of ndarray
            batch size of samples, which might contains the same event multiple times.
        """
        cdef idx = np.random.randint(0,self.get_stored_size(),batch_size)
        return self._encode_sample(idx)

@cython.embedsignature(True)
cdef class ProcessSharedReplayBuffer(ProcessSharedRingEnvironment):
    """
    Replay Buffer class to store environments and to sample them randomly.

    The envitonment contains observation (obs), action (act), reward (rew),
    the next observation (next_obs), and done (done).

    In this class, sampling is random sampling and the same environment can be
    chosen multiple times.

    This class can be add-ed from 'multiprocessing' without explicit lock.
    """
    def __cinit__(self,size,obs_dim=1,act_dim=1,*,rew_dim=1,**kwargs):
        pass

    def __init__(self,size,obs_dim=1,act_dim=1,*,rew_dim=1,**kwargs):
        """
        Parameters
        ----------
        size : int
            buffer size
        obs_dim : int
            observation (obs and next_obs) dimension
        act_dim : int
            action (act) dimension
        rew_dim : int, optional, keyword only
            reward (rew) dimension, whose default value is 1
        """
        pass

    def sample(self,batch_size):
        """
        Sample the stored environment randomly with speciped size

        Parameters
        ----------
        batch_size : int
            sampled batch size

        Returns
        -------
        sample : dict of ndarray
            batch size of samples, which might contains the same event multiple times.
        """
        cdef idx = np.random.randint(0,self.get_stored_size(),batch_size)
        return self._encode_sample(idx)

    def init_worker(self):
        """
        Init worker class for multiprocessing.

        Parameters
        ----------

        Returns
        -------
        ProcessSharedRingEnvironment
            worker class for multiprocessing.


        Notes
        -----
        This method must call in the child process.
        The replay buffer is allocated on shared memory between multi process.
        This function ensure the shared memory address in these pointers.
        """
        return ProcessSharedRingEnvironment(self.buffer_size,
                                            self.obs_dim,self.act_dim,
                                            rew_dim = self.rew_dim,
                                            stored_size = self.stored_size_v,
                                            next_index = self.next_index_v,
                                            obs = self.obs_v,
                                            act = self.act_v,
                                            rew = self.rew_v,
                                            next_obs = self.next_obs_v,
                                            done = self.done_v)

@cython.embedsignature(True)
cdef class SelectiveReplayBuffer(SelectiveEnvironment):
    """
    Replay buffer to store episodes of environment.

    This class can get and delete a episode.
    """
    def __cinit__(self,episode_len,obs_dim=1,act_dim=1,*,Nepisodes=10,rew_dim=1,**kwargs):
        pass

    def __init__(self,episode_len,obs_dim=1,act_dim=1,*,Nepisodes=10,rew_dim=1,**kwargs):
        """
        Parameters
        ----------
        episode_len : int
            the max size of a single episode
        obs_dim : int
            observation (obs, next_obs) dimension
        act_dim : int
            action (act) dimension
        Nepisodes : int, optional
            the max size of stored episodes whose default value is 10
        rew_dim : int, optional
            reward (rew) dimension whose dimension is 1
        """
        pass

    def sample(self,batch_size):
        """
        Sample the stored environment randomly with speciped size

        Parameters
        ----------
        batch_size : int
            sampled batch size

        Returns
        -------
        sample : dict of ndarray
            batch size of samples, which might contains the same event multiple times.
        """
        cdef idx = np.random.randint(0,self.get_stored_size(),batch_size)
        return self._encode_sample(idx)

@cython.embedsignature(True)
cdef class PrioritizedReplayBuffer(RingEnvironment):
    """
    Prioritized replay buffer class to store environments with priorities.
    In this class, these environments are sampled with corresponding priorities.
    """
    cdef VectorDouble weights
    cdef VectorSize_t indexes
    cdef double alpha
    cdef CppPrioritizedSampler[double]* per
    def __cinit__(self,size,obs_dim=1,act_dim=1,*,alpha=0.6,rew_dim=1,**kwrags):
        self.alpha = alpha
        self.per = new CppPrioritizedSampler[double](size,alpha)
        self.weights = VectorDouble()
        self.indexes = VectorSize_t()

    def __init__(self,size,obs_dim=1,act_dim=1,*,alpha=0.6,rew_dim=1,**kwrags):
        """
        Parameters
        ----------
        size : int
            buffer size
        obs_dim : int
            observation (obs, next_obs) dimension
        act_dim : int
            action (act) dimension
        alpha : float, optional
            the exponent of the priorities in stored whose default value is 0.6
        rew_dim : int, optional
            reward (rew) dimension whose default value is 1
        """
        pass

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def add(self,obs,act,rew,next_obs,done,priorities = None):
        """
        Add environment(s) into replay buffer.
        Multiple step environments can be added.

        Parameters
        ----------
        obs : array_like or float or int
            observation(s)
        act : array_like or float or int
            action(s)
        rew : array_like or float or int
            reward(s)
        next_obs : array_like or float or int
            next observation(s)
        done : array_like or float or int
            done(s)
        priorities : array_like or float or int
            priorities of each environment

        Returns
        -------
        int
            the stored first index
        """
        cdef size_t next_index = super().add(obs,act,rew,next_obs,done)
        cdef size_t N = np.array(done,copy=False,ndmin=1).shape[0]
        cdef double [:] ps = np.array(priorities,copy=False,ndmin=1,dtype=np.double)
        if priorities is not None:
            self.per.set_priorities(next_index,&ps[0],N,self.get_buffer_size())
        else:
            self.per.set_priorities(next_index,N,self.get_buffer_size())
        return next_index

    def sample(self,batch_size,beta = 0.4):
        """
        Sample the stored environment depending on correspoinding priorities
        with speciped size

        Parameters
        ----------
        batch_size : int
            sampled batch size
        beta : float, optional
            the exponent for discount priority effect whose default value is 0.4

        Returns
        -------
        sample : dict of ndarray
            batch size of samples which also includes 'weights' and 'indexes'


        Notes
        -----
        When 'beta' is 0, priorities are ignored.
        The greater 'beta', the bigger effect of priories.

        The sampling probabilities are propotional to :math:`priorities ^ {-'beta'}`
        """
        self.per.sample(batch_size,beta,
                        self.weights.vec,self.indexes.vec,
                        self.get_stored_size())
        idx = self.indexes.as_numpy()
        samples = self._encode_sample(idx)
        samples['weights'] = self.weights.as_numpy()
        samples['indexes'] = idx
        return samples

    def update_priorities(self,indexes,priorities):
        """
        Update priorities

        Parameters
        ----------
        indexes : array_like
            indexes to update priorities
        priorities : array_like
            priorities to update

        Returns
        -------
        """
        cdef size_t [:] idx = Csize(indexes)
        cdef double [:] ps = Cview(priorities)
        cdef N = idx.shape[0]
        self.per.update_priorities(&idx[0],&ps[0],N)

    cpdef void clear(self) except *:
        """
        Clear replay buffer

        Parameters
        ----------

        Returns
        -------
        """
        super(PrioritizedReplayBuffer,self).clear()
        clear(self.per)

    cpdef double get_max_priority(self):
        """
        Get the max priority of stored priorities

        Parameters
        ----------

        Returns
        -------
        max_priority : double
            the max priority of stored priorities
        """
        return self.per.get_max_priority()

@cython.embedsignature(True)
cdef class ProcessSharedPrioritizedWorker(ProcessSharedRingEnvironment):
    """
    Worker class for PrioritizedReplayBuffer
    """
    cdef VectorDouble weights
    cdef VectorSize_t indexes
    cdef double alpha
    cdef CppThreadSafePrioritizedSampler[double]* per
    cdef max_priority
    cdef sum_tree
    cdef sum_anychanged
    cdef sum_changed
    cdef min_tree
    cdef min_anychanged
    cdef min_changed
    def __cinit__(self,size,obs_dim=1,act_dim=1,*,alpha=0.6,rew_dim=1,
                  max_priority = None,
                  sum_tree = None,sum_anychanged = None,sum_changed = None,
                  min_tree = None,min_anychanged = None,min_changed = None,
                  initialize = True,**kwrags):
        cdef N = self.buffer_size

        self.alpha = alpha

        self.max_priority   = max_priority   or RawArray(ctypes.c_double,1)
        self.sum_tree       = sum_tree       or RawArray(ctypes.c_double,2*N-1)
        self.sum_anychanged = sum_anychanged or RawArray(ctypes.c_char  ,1)
        self.sum_changed    = sum_changed    or RawArray(ctypes.c_char  ,N)
        self.min_tree       = min_tree       or RawArray(ctypes.c_double,2*N-1)
        self.min_anychanged = min_anychanged or RawArray(ctypes.c_char  ,1)
        self.min_changed    = min_changed    or RawArray(ctypes.c_char  ,N)

        cdef double [:] max_priority_view = self.max_priority
        cdef double [:] sum_tree_view = self.sum_tree
        cdef bool [:] sum_anychanged_view = self.sum_anychanged
        cdef bool [:] sum_changed_view = self.sum_changed
        cdef double [:] min_tree_view = self.min_tree
        cdef bool [:] min_anychanged_view = self.min_anychanged
        cdef bool [:] min_changed_view = self.min_changed

        self.per= new CppThreadSafePrioritizedSampler[double](N,alpha,
                                                              &max_priority_view[0],
                                                              &sum_tree_view[0],
                                                              &sum_anychanged_view[0],
                                                              &sum_changed_view[0],
                                                              &min_tree_view[0],
                                                              &min_anychanged_view[0],
                                                              &min_changed_view[0],
                                                              initialize)
        self.weights = VectorDouble()
        self.indexes = VectorSize_t()

    def __init__(self,size,obs_dim=1,act_dim=1,*,alpha=0.6,rew_dim=1,
                  max_priority = None,
                  sum_tree = None,sum_anychanged = None,sum_changed = None,
                  min_tree = None,min_anychanged = None,min_changed = None,
                  initialize = True,**kwrags):
        """
        Parameters
        ----------
        size : int
            buffer size
        obs_dim : int
            observation (obs, next_obs) dimension
        act_dim : int
            action (act) dimension
        alpha : float, optional
            the exponent of the stored priorities whose default value is 0.6
        rew_dim : int, optional
            reward (rew) dimension whose default value is 1
        max_priority : multiprocessing.RawArray of ctypes.c_double
            shared memory for max_priority. If None, new memory is allocated.
        sum_tree : multiprocessing.RawArray of ctypes.c_double
            shared memory for sum_tree. If None, new memory is allocated.
        sum_anychanged : multiprocessing.RawArray of ctypes.c_char
            shared memory for sum_anychanged. If None, new memory is allocated.
        sum_changed : multiprocessing.RawArray of ctypes.c_char
            shared memory for sum_changed. If None, new memory is allocated.
        min_tree : multiprocessing.RawArray of ctypes.c_double
            shared memory for min_tree. If None, new memory is allocated.
        min_anychanged : multiprocessing.RawArray of ctypes.c_char
            shared memory for min_anychanged. If None, new memory is allocated.
        min_changed : multiprocessing.RawArray of ctypes.c_char
            shared memory for min_changed. If None, new memory is allocated.
        initialize : bool
            flag for initialization whose default value is True. If True, initialize.
        """
        pass

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def add(self,obs,act,rew,next_obs,done,priorities = None):
        """
        Add environment(s) into replay buffer.
        Multiple step environments can be added.

        Parameters
        ----------
        obs : array_like or float or int
            observation(s)
        act : array_like or float or int
            action(s)
        rew : array_like or float or int
            reward(s)
        next_obs : array_like or float or int
            next observation(s)
        done : array_like or float or int
            done(s)
        priorities : array_like or float or int
            priorities of each environment

        Returns
        -------
        int
            the stored first index
        """
        cdef size_t next_index = super().add(obs,act,rew,next_obs,done)
        cdef size_t N = np.array(done,copy=False,ndmin=1).shape[0]
        cdef double [:] ps = Cview(priorities)
        if priorities is not None:
            self.per.set_priorities(next_index,&ps[0],N,self.get_buffer_size())
        else:
            self.per.set_priorities(next_index,N,self.get_buffer_size())
        return next_index

    def update_priorities(self,indexes,priorities):
        """
        Update priorities

        Parameters
        ----------
        indexes : array_like
            indexes to update priorities
        priorities : array_like
            priorities to update

        Returns
        -------
        """
        cdef size_t [:] idx = Csize(indexes)
        cdef double [:] ps = Cview(priorities)
        cdef size_t N = idx.shape[0]
        self.per.update_priorities(&idx[0],&ps[0],N)

    cpdef void clear(self) except *:
        """
        Clear replay buffer

        Parameters
        ----------

        Returns
        -------
        """
        super(ProcessSharedPrioritizedWorker,self).clear()
        clear(self.per)

    cpdef double get_max_priority(self):
        """
        Get the max priority of stored priorities

        Parameters
        ----------

        Returns
        -------
        max_priority : double
            the max priority of stored priorities
        """
        return self.per.get_max_priority()

@cython.embedsignature(True)
cdef class ProcessSharedPrioritizedReplayBuffer(ProcessSharedPrioritizedWorker):
    """
    Prioritized replay buffer class to store environments with priorities.
    In this class, these environments are sampled with corresponding priorities.

    This class can be added from multiprocessing without explicit lock.
    """
    def sample(self,batch_size,beta = 0.4):
        """
        Sample the stored environment depending on correspoinding priorities
        with speciped size

        Parameters
        ----------
        batch_size : int
            sampled batch size
        beta : float, optional
            the exponent for discount priority effect whose default value is 0.4

        Returns
        -------
        sample : dict of ndarray
            batch size of samples which also includes 'weights' and 'indexes'


        Notes
        -----
        When 'beta' is 0, priorities are ignored.
        The greater 'beta', the bigger effect of priories.

        The sampling probabilities are propotional to :math:`priorities ^ {-'beta'}`
        """
        self.per.sample(batch_size,beta,
                        self.weights.vec,self.indexes.vec,
                        self.get_stored_size())
        idx = self.indexes.as_numpy()
        samples = self._encode_sample(idx)
        samples['weights'] = self.weights.as_numpy()
        samples['indexes'] = idx
        return samples

    def init_worker(self):
        """
        Init worker class for multiprocessing.

        Parameters
        ----------

        Returns
        -------
        ProcessSharedPrioritizedWorker
            worker class for multiprocessing.


        Notes
        -----
        This method must call in the child process.
        The replay buffer is allocated on shared memory between multi process.
        This function ensure the shared memory address in these pointers.
        """
        return ProcessSharedPrioritizedWorker(self.buffer_size,
                                              self.obs_dim,self.act_dim,
                                              rew_dim = self.rew_dim,
                                              alpha = self.alpha,
                                              stored_size = self.stored_size_v,
                                              next_index = self.next_index_v,
                                              obs = self.obs_v,
                                              act = self.act_v,
                                              rew = self.rew_v,
                                              next_obs = self.next_obs_v,
                                              done = self.done_v,
                                              max_priority = self.max_priority,
                                              sum_tree = self.sum_tree,
                                              sum_anychanged = self.sum_anychanged,
                                              sum_changed = self.sum_changed,
                                              min_tree = self.min_tree,
                                              min_anychanged = self.min_anychanged,
                                              min_changed = self.min_changed,
                                              initialize = False)


@cython.embedsignature(True)
cdef class NstepReplayBuffer(ReplayBuffer):
    """
    Replay buffer using N-step reward
    """
    cdef CppNstepRewardBuffer[double,double]* nrb
    cdef PointerDouble gamma
    cdef PointerDouble nstep_rew
    cdef PointerDouble nstep_next_obs
    def __cinit__(self,size,obs_dim=1,act_dim=1,*,n_step = 4, discount = 0.99,**kwargs):
        self.nrb = new CppNstepRewardBuffer[double,double](size,obs_dim,
                                                           n_step,discount)
        self.gamma = PointerDouble(ndim=2,value_dim=1,size=size)
        self.nstep_rew = PointerDouble(ndim=2,value_dim=1,size=size)
        self.nstep_next_obs = PointerDouble(ndim=2,value_dim=obs_dim,size=size)

    def __init__(self,size,obs_dim=1,act_dim=1,*,n_step = 4, discount = 0.99,**kwargs):
        """
        Parameters
        ----------
        size : int
            buffer size
        obs_dim : int
            observation (obs, next_obs) dimension
        act_dim : int
            action (act) dimension
        n_step : int, optional
            the number of n-step reward whose default value is 4
        discount : float, optional
            discount factor for reward
        """
        pass

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
        samples['discounts'] = self.gamma.as_numpy()
        samples['rew'] = self.nstep_rew.as_numpy()
        samples['next_obs'] = self.nstep_next_obs.as_numpy()
        return samples

@cython.embedsignature(True)
cdef class NstepPrioritizedReplayBuffer(PrioritizedReplayBuffer):
    """
    Prioritiezed replay buffer using N-step reward
    """
    cdef CppNstepRewardBuffer[double,double]* nrb
    cdef PointerDouble gamma
    cdef PointerDouble nstep_rew
    cdef PointerDouble nstep_next_obs
    def __cinit__(self,size,obs_dim=1,act_dim=1,*,
                  alpha = 0.6,n_step = 4, discount = 0.99,**kwargs):
        self.nrb = new CppNstepRewardBuffer[double,double](size,obs_dim,
                                                           n_step,discount)
        self.gamma = PointerDouble(ndim=2,value_dim=1,size=size)
        self.nstep_rew = PointerDouble(ndim=2,value_dim=1,size=size)
        self.nstep_next_obs = PointerDouble(ndim=2,value_dim=obs_dim,size=size)

    def __init__(self,size,obs_dim=1,act_dim=1,*,
                 alpha = 0.6,n_step = 4, discount = 0.99,**kwargs):
        """
        Parameters
        ----------
        size : int
            buffer size
        obs_dim : int
            observation (obs, next_obs) dimension
        act_dim : int
            action (act) dimension
        alpha : float, optional
            the exponent of stored priorities whose default value is 0.6
        discount : float
            discount factor for N-step reward whose default value is 0.99
        """
        pass
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
        samples['discounts'] = self.gamma.as_numpy()
        samples['rew'] = self.nstep_rew.as_numpy()
        samples['next_obs'] = self.nstep_next_obs.as_numpy()
        return samples

def create_buffer(size,obs_dim=1,act_dim=1,*,
                  prioritized = False, Nstep = False, process_shared = False,**kwarg):
    """Create specified version of replay buffer

    Parameters
    ----------
    size: int
        buffer size
    obs_dim: int
        observation (obs, next_obs) dimension
    act_dim: int
        action (act) dimension
    prioritized: bool, optional
        create prioritized version replay buffer, default = False
    Nstep: bool, optional
        create Nstep version replay buffer, default = False
    process_shared: bool
        create process shared version replay buffer, default = False

    Returns
    -------
    : one of the replay buffer classes

    Raises
    ------
    NotImplementedError
        If you specified not implemented version replay buffer

    Note
    ----
    Any other keyword arguments are passed to replay buffer constructor.

    """
    per = "Prioritized" if prioritized else ""
    nstep = "Nstep" if Nstep else ""
    ps = "ProcessShared" if process_shared else ""

    buffer_name = f"{ps}{nstep}{per}ReplayBuffer"

    cls={"ReplayBuffer": ReplayBuffer,
         "PrioritizedReplayBuffer": PrioritizedReplayBuffer,
         "NstepReplayBuffer": NstepReplayBuffer,
         "NstepPrioritizedReplayBuffer": NstepPrioritizedReplayBuffer,
         "ProcessSharedReplayBuffer": ProcessSharedReplayBuffer,
         "ProcessSharedPrioritizedReplayBuffer": ProcessSharedPrioritizedReplayBuffer}

    buffer = cls.get(f"{buffer_name}",None)

    if buffer:
        return buffer(size,obs_dim=obs_dim,act_dim=act_dim,**kwarg)

    raise NotImplementedError(f"{buffer_name} is not Implemented")

@cython.boundscheck(False)
@cython.wraparound(False)
def explore(buffer,policy,env,n_iteration,*,
            local_buffer = 10,longest_step = 500,rew_func = None, callback = None):
    """Explore multiple iterations and add envitonment into buffer

    Parameters
    ----------
    buffer: one of the ReplayBuffer classes
        replay buffer to be stored
    policy: callable
        policy returns action from observation
    env: gym.Env
        gym.Env compatible environment
    local_buffer: int, optional
        local buffer size before adding replay buffer. default = 10
    longest_step: int
        longest step in a single episode. default = 500
    rew_func: callable
        function to custom reward from obs, act, rew, next_obs, and doe
    callback: callable
        call-back function to call at each step
    """
    cdef size_t ITERATION = n_iteration
    cdef size_t LOCAL = local_buffer
    cdef size_t LONGEST = longest_step

    obs = np.zeros((LOCAL,buffer.get_obs_dim()),dtype=np.double)
    next_obs = np.zeros_like(obs)
    act = np.zeros((LOCAL,buffer.get_act_dim()),dtype=np.double)
    rew = np.zeros((LOCAL,buffer.get_rew_dim()),dtype=np.double)
    done = np.zeros((LOCAL),dtype=np.double)

    cdef double [:,:] o = obs
    cdef double [:,:] a = act
    cdef double [:,:] r = rew
    cdef double [:,:] no= next_obs
    cdef double [:]  d = done

    cdef double [:] _o
    cdef double [:] _a
    cdef double [:] _r

    cdef size_t it
    cdef size_t idx = 0
    cdef size_t step = 0
    cdef size_t tmp_i

    cdef bool custom_rew = rew_func
    cdef bool use_callback = callback

    for it in range(ITERATION):
        _o = env.reset()
        o[idx] = _o

        for step in range(LONGEST):
            _a = policy(o[idx])
            a[idx] = _a

            next_obs[idx], rew[idx], done[idx], _ = env.step(a[idx])
            # Use ndarray for unpack assignment because of cython bug.
            # https://github.com/cython/cython/issues/541

            if custom_rew:
                _r = rew_func(obs=o[idx], act=a[idx], rew=r[idx],
                              next_obs=no[idx], done=d[idx])
                r[idx] = _r

            if use_callback:
                callback(obs=o[idx], act=a[idx], rew=r[idx],
                         next_obs=no[idx], done=d[idx],
                         step = step, iteration = it)

            tmp_i = idx + 1
            if tmp_i == LOCAL:
                tmp_i = 0
                buffer.add(o,a,r,no,d)

            if d[idx]:
                idx = tmp_i
                break

            o[tmp_i] = no[idx]
            idx = tmp_i

    if idx != 0:
        buffer.add(o[:idx],a[:idx],r[:idx],no[:idx],d[:idx])
