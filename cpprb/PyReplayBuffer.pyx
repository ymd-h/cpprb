# distutils: language = c++
# cython: linetrace=True

import ctypes
cimport numpy as np
import numpy as np
import cython

from cpprb.ReplayBuffer cimport *

from .VectorWrapper cimport *
from .VectorWrapper import (VectorWrapper,
                            VectorInt,VectorSize_t,
                            VectorDouble,PointerDouble,VectorFloat)

cdef double [::1] Cdouble(array):
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
        obs_dim : int, optional
            observation (obs) dimension whose default value is 1
        act_dim : int, optional
            action (act) dimension whose default value is 1
        rew_dim : int, optional
            reward (rew) dimension whose default value is 1
        is_discrete_action: bool, optional
            If True, act_dim is compressed to 1 whose default value is False
        obs_shape: array-like
            observation shape. If not None, overwrite obs_dim.
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
        return self._add(Cdouble(obs),Cdouble(act),Cdouble(rew),Cdouble(next_obs),Cdouble(done))

    def _encode_sample(self,idx):
        dtype = np.int if self.is_discrete_action else np.double

        _o = self.obs.as_numpy()[idx]
        _no = self.next_obs.as_numpy()[idx]
        if self.obs_shape is not None:
            _shape = (-1,*self.obs_shape)
            _o = _o.reshape(_shape)
            _no = _no.reshape(_shape)

        return {'obs': _o,
                'act': self.act.as_numpy(dtype=dtype)[idx],
                'rew': self.rew.as_numpy()[idx],
                'next_obs': _no,
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

    def get_obs_shape(self):
        """Return observation shape
        """
        return self.obs_shape

    cpdef size_t get_act_dim(self):
        """Return action dimension (act_dim)
        """
        return self.act_dim

    cpdef size_t get_rew_dim(self):
        """Return reward dimension (rew_dim)
        """
        return self.rew_dim


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
                                                                 self.obs_dim,
                                                                 self.act_dim,
                                                                 self.rew_dim)

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
