# distutils: language = c++
# cython: linetrace=True

import ctypes
from logging import getLogger, StreamHandler, Formatter, INFO
from multiprocessing import Event, Lock, Process
from multiprocessing.sharedctypes import Value, RawValue, RawArray
import time
from typing import Any, Dict, Callable, Optional
import warnings

cimport numpy as np
import numpy as np
import cython
from cython.operator cimport dereference

from cpprb.ReplayBuffer cimport *

from .VectorWrapper cimport *
from .VectorWrapper import (VectorWrapper,
                            VectorInt,VectorSize_t,
                            VectorDouble,PointerDouble,VectorFloat)

def default_logger(level=INFO):
    """
    Create default logger for cpprb
    """
    logger = getLogger("cpprb")
    logger.setLevel(level)

    handler = StreamHandler()
    handler.setLevel(level)

    format = Formatter("%(asctime)s.%(msecs)03d [%(levelname)s] " +
                       "(%(filename)s:%(lineno)s) %(message)s",
                       "%Y%m%d-%H%M%S")
    handler.setFormatter(format)

    if logger.hasHandlers():
        logger.handlers[0] = handler
    else:
        logger.addHandler(handler)
        logger.propagate = False

    return logger

cdef double [::1] Cdouble(array):
    return np.ravel(np.array(array,copy=False,dtype=np.double,ndmin=1,order='C'))

cdef inline const size_t [::1] Csize(array):
    return np.ravel(np.array(array,copy=False,dtype=np.uint64,ndmin=1,order='C'))

@cython.embedsignature(True)
cdef inline const float [::1] Cfloat(array):
    return np.ravel(np.array(array,copy=False,dtype=np.single,ndmin=1,order='C'))


def unwrap(d):
    return d[np.newaxis][0]


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


cdef class SharedBuffer:
    cdef dtype
    cdef data
    cdef data_ndarray
    cdef view
    def __init__(self,shape,dtype,data=None):
        self.dtype = np.dtype(dtype)

        if data is None:
            try:
                ctype = np.ctypeslib.as_ctypes_type(self.dtype)
            except NotImplementedError:
                # Dirty hack to allocate correct size shared memory
                for d in (np.int8, np.int16, np.int32, np.int64):
                    _d = np.dtype(d)

                    if self.dtype.itemsize == _d.itemsize:
                        ctype = np.ctypeslib.as_ctypes_type(_d)
                        break
                else:
                    raise

            len = int(np.array(shape,copy=False,dtype="int").prod())
            self.data = RawArray(ctype,len)
        else:
            self.data = data

        self.data_ndarray = np.ctypeslib.as_array(self.data)
        self.data_ndarray.shape = shape

        # Reinterpretation
        if self.dtype != self.data_ndarray.dtype:
            self.view = self.data_ndarray.view(self.dtype)
        else:
            self.view = self.data_ndarray


    def __getitem__(self,key):
        return self.view[key]

    def __setitem__(self,key,value):
        self.view[key] = value

    def __reduce__(self):
        return (SharedBuffer,(self.view.shape,self.dtype,self.data))


def dict2buffer(buffer_size: int,env_dict: Dict,*,
                stack_compress = None, default_dtype = None,
                mmap_prefix: Optional[str] = None,
                shared: bool = False):
    """Create buffer from env_dict

    Parameters
    ----------
    buffer_size : int
        buffer size
    env_dict : dict of dict
        Specify environment values to be stored in buffer.
    stack_compress : str or array like of str, optional
        compress memory of specified stacked values.
    default_dtype : numpy.dtype, optional
        fallback dtype for not specified in `env_dict`. default is numpy.single
    mmap_prefix : str, optional
        File name prefix to save buffer data using mmap. If `None` (default),
        save only on memory.

    Returns
    -------
    buffer : dict of numpy.ndarray
        buffer for environment specified by env_dict.
    """
    cdef buffer = {}
    cdef bool compress_any = stack_compress
    default_dtype = default_dtype or np.single

    def zeros(name,shape,dtype):
        if shared:
            return SharedBuffer(shape,dtype)

        if mmap_prefix:
            if not isinstance(shape,tuple):
                shape = tuple(shape)
            return np.memmap(f"{mmap_prefix}_{name}.dat",
                             shape=shape,dtype=dtype,mode="w+")
        else:
            return np.zeros(shape=shape,dtype=dtype)

    for name, defs in env_dict.items():
        shape = np.insert(np.asarray(defs.get("shape",1)),0,buffer_size)

        if compress_any and np.isin(name,
                                    stack_compress,
                                    assume_unique=True).any():
            buffer_shape = np.insert(np.delete(shape,-1),1,shape[-1])
            buffer_shape[0] += buffer_shape[1] - 1
            buffer_shape[1] = 1
            memory = zeros(name, buffer_shape,
                           dtype=defs.get("dtype",default_dtype))
            strides = np.append(np.delete(memory.strides,1),memory.strides[1])
            buffer[name] = np.lib.stride_tricks.as_strided(memory,
                                                           shape=shape,
                                                           strides=strides)
        else:
            buffer[name] = zeros(name,shape,dtype=defs.get("dtype",default_dtype))

        buffer[name][:] = 1

        shape[0] = -1
        defs["add_shape"] = shape
    return buffer

def find_array(dict,key):
    """Find 'key' and ensure numpy.ndarray with the minimum dimension of 1.

    Parameters
    ----------
    dict : dict
        dict where find 'key'
    key : str
        dictionary key to find

    Returns
    -------
    : numpy.ndarray or None
        If `dict` has `key`, returns the values with numpy.ndarray with the minimum
        dimension of 1. Otherwise, returns `None`.
    """
    return None if not key in dict else np.array(dict[key],ndmin=1,copy=False)

@cython.embedsignature(True)
cdef class StepChecker:
    """Check the step size of addition
    """
    cdef check_str
    cdef check_shape

    def __init__(self,env_dict,special_keys = None):
        """Initialize StepChecker class.

        Parameters
        ----------
        env_dict : dict
            Specify the environment values.
        """
        special_keys = special_keys or []
        for name, defs in env_dict.items():
            if name in special_keys:
                continue
            self.check_str = name
            self.check_shape = defs["add_shape"]

    cdef size_t step_size(self,kwargs) except *:
        """Return step size.

        Parameters
        ----------
        kwargs: dict
            Added values.
        """
        return np.reshape(kwargs[self.check_str],self.check_shape,order='A').shape[0]

@cython.embedsignature(True)
cdef class NstepBuffer:
    """Local buffer class for Nstep reward.

    This buffer temporary stores environment values and returns Nstep-modified
    environment values for `ReplayBuffer`
    """
    cdef buffer
    cdef size_t buffer_size
    cdef default_dtype
    cdef size_t stored_size
    cdef size_t Nstep_size
    cdef float Nstep_gamma
    cdef Nstep_rew
    cdef Nstep_next
    cdef env_dict
    cdef stack_compress
    cdef StepChecker size_check

    def __cinit__(self,env_dict=None,Nstep=None,*,
                  stack_compress = None,default_dtype = None,next_of = None):
        self.env_dict = env_dict.copy() if env_dict else {}
        self.stored_size = 0
        self.stack_compress = None # stack_compress is not support yet.
        self.default_dtype = default_dtype or np.single

        if next_of is not None: # next_of is not support yet.
            for name in np.array(next_of,copy=False,ndmin=1):
                self.env_dict[f"next_{name}"] = self.env_dict[name]

        self.Nstep_size = Nstep["size"]
        self.Nstep_gamma = Nstep.get("gamma",0.99)
        self.Nstep_rew = find_array(Nstep,"rew")
        self.Nstep_next = find_array(Nstep,"next")

        self.buffer_size = self.Nstep_size - 1
        self.buffer = dict2buffer(self.buffer_size,self.env_dict,
                                  stack_compress = self.stack_compress,
                                  default_dtype = self.default_dtype)
        self.size_check = StepChecker(self.env_dict)

    def __init__(self,env_dict=None,Nstep=None,*,
                 stack_compress = None,default_dtype = None, next_of = None):
        r"""Initialize NstepBuffer class.

        Parameters
        ----------
        env_dict : dict
            Specify environment values to be stored.
        Nstep : dict
            `Nstep["size"]` is `int` specifying step size of Nstep reward.
            `Nstep["rew"]` is `str` or array like of `str` specifying
            Nstep reward to be summed. `Nstep["gamma"]` is float specifying
            discount factor, its default is 0.99. `Nstep["next"]` is `str` or
            list of `str` specifying next values to be moved.
        stack_compress : str or array like of str, optional
            compress memory of specified stacked values.
        default_dtype : numpy.dtype, optional
            fallback dtype for not specified in `env_dict`. default is numpy.single
        next_of : str or array like of str, optional
            next item of specified environemt variables (eg. next_obs for next) are
            also sampled without duplicated values

        Notes
        -----
        Currently, memory compression features (`stack_compress` and `next_of`) are
        not supported yet. (Fall back to usual storing)
        """
        pass

    def add(self,*,**kwargs):
        r"""Add envronment into local buffer.

        Paremeters
        ----------
        **kwargs : keyword arguments
            Values to be added.

        Returns
        -------
        env : dict or None
            Values with Nstep reward calculated. When the local buffer does not
            store enough cache items, returns 'None'.
        """
        cdef size_t N = self.size_check.step_size(kwargs)
        cdef ssize_t end = self.stored_size + N

        cdef ssize_t i
        cdef ssize_t stored_begin
        cdef ssize_t stored_end
        cdef ssize_t ext_begin
        cdef ssize_t max_slide

        # Case 1
        #   If Nstep buffer don't become full, store all the input transitions.
        #   These transitions are partially calculated.
        if end <= self.buffer_size:
            for name, stored_b in self.buffer.items():
                if self.Nstep_rew is not None and np.isin(name,self.Nstep_rew).any():
                    # Calculate later.
                    pass
                elif (self.Nstep_next is not None
                      and np.isin(name,self.Nstep_next).any()):
                    # Do nothing.
                    pass
                else:
                    stored_b[self.stored_size:end] = self._extract(kwargs,name)

            # Nstep reward must be calculated after "done" filling
            gamma = (1.0 - self.buffer["done"][:end]) * self.Nstep_gamma

            if self.Nstep_rew is not None:
                max_slide = min(self.Nstep_size - self.stored_size,N)
                max_slide *= -1
                for name in self.Nstep_rew:
                    ext_b = self._extract(kwargs,name).copy()
                    self.buffer[name][self.stored_size:end] = ext_b

                    for i in range(self.stored_size-1,max_slide,-1):
                        stored_begin = max(i,0)
                        stored_end = i+N
                        ext_begin = max(-i,0)
                        ext_b[ext_begin:] *= gamma[stored_begin:stored_end]
                        self.buffer[name][stored_begin:stored_end] +=ext_b[ext_begin:]

            self.stored_size = end
            return None

        # Case 2
        #   If we have enough transitions, return calculated transtions
        cdef size_t diff_N = self.buffer_size - self.stored_size
        cdef size_t add_N = N - diff_N
        cdef bool NisBigger = (add_N > self.buffer_size)
        end = self.buffer_size if NisBigger else add_N

        # Nstep reward must be calculated before "done" filling
        cdef ssize_t spilled_N
        gamma = np.ones((self.stored_size + N,1),dtype=np.single)
        gamma[:self.stored_size] -= self.buffer["done"][:self.stored_size]
        gamma[self.stored_size:] -= self._extract(kwargs,"done")
        gamma *= self.Nstep_gamma
        if self.Nstep_rew is not None:
            max_slide = min(self.Nstep_size - self.stored_size,N)
            max_slide *= -1
            for name in self.Nstep_rew:
                stored_b = self.buffer[name]
                ext_b = self._extract(kwargs,name)

                copy_ext = ext_b.copy()
                if diff_N:
                    stored_b[self.stored_size:] = ext_b[:diff_N]
                    ext_b = ext_b[diff_N:]

                for i in range(self.stored_size-1,max_slide,-1):
                    stored_begin = max(i,0)
                    stored_end = i+N
                    ext_begin = max(-i,0)
                    copy_ext[ext_begin:] *= gamma[stored_begin:stored_end]
                    if stored_end <= self.buffer_size:
                        stored_b[stored_begin:stored_end] += copy_ext[ext_begin:]
                    else:
                        spilled_N = stored_end - self.buffer_size
                        stored_b[stored_begin:] += copy_ext[ext_begin:-spilled_N]
                        ext_b[:spilled_N] += copy_ext[-spilled_N:]

                self._roll(stored_b,ext_b,end,NisBigger,kwargs,name,add_N)

        for name, stored_b in self.buffer.items():
            if self.Nstep_rew is not None and np.isin(name,self.Nstep_rew).any():
                # Calculated.
                pass
            elif (self.Nstep_next is not None
                  and np.isin(name,self.Nstep_next).any()):
                kwargs[name] = self._extract(kwargs,name)[diff_N:]
            else:
                ext_b = self._extract(kwargs,name)

                if diff_N:
                    stored_b[self.stored_size:] = ext_b[:diff_N]
                    ext_b = ext_b[diff_N:]

                self._roll(stored_b,ext_b,end,NisBigger,kwargs,name,add_N)

        done = kwargs["done"]

        for i in range(1,self.buffer_size):
            if i <= add_N:
                done[:-i] += kwargs["done"][i:]
                done[-i:] += self.buffer["done"][:i]
            else:
                done += self.buffer["done"][i-add_N:i]

        self.stored_size = self.buffer_size
        return kwargs

    cdef _extract(self,kwargs,name):
        _dict = self.env_dict[name]
        return np.reshape(np.array(kwargs[name],copy=False,ndmin=2,
                                   dtype=_dict.get("dtype",self.default_dtype)),
                          _dict["add_shape"])

    cdef void _roll(self,stored_b,ext_b,
                    ssize_t end,bool NisBigger,kwargs,name,size_t add_N):
        # Swap numpy.ndarray
        # https://stackoverflow.com/a/33362030
        stored_b[:end], ext_b[-end:] = ext_b[-end:], stored_b[:end].copy()
        if NisBigger:
            # buffer: XXXX, add: YYYYY
            # buffer: YYYY, add: YXXXX
            ext_b = np.roll(ext_b,end,axis=0)
            # buffer: YYYY, add: XXXXY
        else:
            # buffer: XXXZZZZ, add: YYY
            # buffer: YYYZZZZ, add: XXX
            stored_b[:] = np.roll(stored_b,-end,axis=0)[:]
            # buffer: ZZZZYYY, add: XXX
        kwargs[name] = ext_b[:add_N]

    cpdef void clear(self):
        """Clear the bufer.
        """
        self.stored_size = 0

    cpdef on_episode_end(self):
        """Terminate episode.
        """
        kwargs = {k: v[:self.stored_size].copy() for k, v in self.buffer.items()}
        done = kwargs["done"]

        for i in range(1,self.stored_size):
            done[:-i] += kwargs["done"][i:]

        self.clear()
        return kwargs

    cpdef size_t get_Nstep_size(self):
        """Get Nstep size

        Returns
        -------
        Nstep_size : size_t
            Nstep size
        """
        return self.Nstep_size


cdef class RingBufferIndex:
    """Ring Buffer Index class
    """
    cdef index
    cdef buffer_size
    cdef is_full

    def __init__(self,buffer_size):
        self.index = RawValue(ctypes.c_size_t,0)
        self.buffer_size = RawValue(ctypes.c_size_t,buffer_size)
        self.is_full = RawValue(ctypes.c_int,0)

    cdef size_t get_next_index(self):
        return self.index.value

    cdef size_t fetch_add(self,size_t N):
        """
        Add then return original value

        Parameters
        ----------
        N : size_t
            value to add

        Returns
        -------
        size_t
            index before add
        """
        cdef size_t ret = self.index.value
        self.index.value += N

        if self.index.value >= self.buffer_size.value:
            self.is_full.value = 1

        while self.index.value >= self.buffer_size.value:
            self.index.value -= self.buffer_size.value

        return ret

    cdef void clear(self):
        self.index.value = 0
        self.is_full.value = 0

    cdef size_t get_stored_size(self):
        if self.is_full.value:
            return self.buffer_size.value
        else:
            return self.index.value


cdef class ProcessSafeRingBufferIndex(RingBufferIndex):
    """Process Safe Ring Buffer Index class
    """
    cdef lock

    def __init__(self,buffer_size):
        super().__init__(buffer_size)
        self.lock = Lock()

    cdef size_t get_next_index(self):
        with self.lock:
            return RingBufferIndex.get_next_index(self)

    cdef size_t fetch_add(self,size_t N):
        with self.lock:
            return RingBufferIndex.fetch_add(self,N)

    cdef void clear(self):
        with self.lock:
            RingBufferIndex.clear(self)

    cdef size_t get_stored_size(self):
        with self.lock:
            return RingBufferIndex.get_stored_size(self)


@cython.embedsignature(True)
cdef class ReplayBuffer:
    r"""Replay Buffer class to store transitions and to sample them randomly.

    The transition can contain anything compatible with numpy data
    type. User can specify by `env_dict` parameters at constructor
    freely.

    The possible standard transition contains observation (`obs`), action (`act`),
    reward (`rew`), the next observation (`next_obs`), and done (`done`).

    >>> env_dict = {"obs": {"shape": (4,4)},
                    "act": {"shape": 3, "dtype": np.int16},
                    "rew": {},
                    "next_obs": {"shape": (4,4)},
                    "done": {}}

    In this class, sampling is random sampling and the same transition
    can be chosen multiple times."""
    cdef buffer
    cdef size_t buffer_size
    cdef env_dict
    cdef RingBufferIndex index
    cdef size_t episode_len
    cdef next_of
    cdef bool has_next_of
    cdef next_
    cdef bool compress_any
    cdef stack_compress
    cdef cache
    cdef default_dtype
    cdef StepChecker size_check
    cdef NstepBuffer nstep
    cdef bool use_nstep
    cdef size_t cache_size

    def __cinit__(self,size,env_dict=None,*,
                  next_of=None,stack_compress=None,default_dtype=None,Nstep=None,
                  mmap_prefix =None,
                  **kwargs):
        self.env_dict = env_dict.copy() if env_dict else {}
        cdef special_keys = []

        self.buffer_size = size
        self.index = RingBufferIndex(self.buffer_size)
        self.episode_len = 0

        self.compress_any = stack_compress
        self.stack_compress = np.array(stack_compress,ndmin=1,copy=False)

        self.default_dtype = default_dtype or np.single

        self.has_next_of = next_of
        self.next_of = np.array(next_of,
                                ndmin=1,copy=False) if self.has_next_of else None
        self.next_ = {}
        self.cache = {} if (self.has_next_of or self.compress_any) else None

        self.use_nstep = Nstep
        if self.use_nstep:
            self.nstep = NstepBuffer(self.env_dict,Nstep.copy(),
                                     stack_compress = self.stack_compress,
                                     next_of = self.next_of,
                                     default_dtype = self.default_dtype)

            # Nstep is not support next_of yet
            self.next_of = None
            self.has_next_of = False

        # side effect: Add "add_shape" key into self.env_dict
        self.buffer = dict2buffer(self.buffer_size,self.env_dict,
                                  stack_compress = self.stack_compress,
                                  default_dtype = self.default_dtype,
                                  mmap_prefix = mmap_prefix)

        self.size_check = StepChecker(self.env_dict,special_keys)

        # Cache Size:
        #     No "next_of" nor "stack_compress": -> 0
        #     If "stack_compress": -> max of stack size -1
        #     If "next_of": -> Increase by 1
        self.cache_size = 1 if (self.cache is not None) else 0
        if self.compress_any:
            for name in self.stack_compress:
                self.cache_size = max(self.cache_size,
                                      np.array(self.env_dict[name]["shape"],
                                               ndmin=1,copy=False)[-1] -1)

        if self.has_next_of:
            self.cache_size += 1
            for name in self.next_of:
                self.next_[name] = self.buffer[name][0].copy()

    def __init__(self,size,env_dict=None,*,
                 next_of=None,stack_compress=None,default_dtype=None,Nstep=None,
                 mmap_prefix =None,
                 **kwargs):
        r"""Initialize ReplayBuffer

        Parameters
        ----------
        size : int
            buffer size
        env_dict : dict of dict, optional
            dictionary specifying environments. The keies of env_dict become
            environment names. The values of env_dict, which are also dict,
            defines "shape" (default 1) and "dtypes" (fallback to `default_dtype`)
        next_of : str or array like of str, optional
            next item of specified environemt variables (eg. next_obs for next) are
            also sampled without duplicated values
        stack_compress : str or array like of str, optional
            compress memory of specified stacked values.
        default_dtype : numpy.dtype, optional
            fallback dtype for not specified in `env_dict`. default is numpy.single
        Nstep : dict, optional
            `Nstep["size"]` is `int` specifying step size of Nstep reward.
            `Nstep["rew"]` is `str` or array like of `str` specifying
            Nstep reward to be summed. `Nstep["gamma"]` is float specifying
            discount factor, its default is 0.99. `Nstep["next"]` is `str` or
            list of `str` specifying next values to be moved.
        mmap_prefix : str, optional
            File name prefix to save buffer data using mmap. If `None` (default),
            save only on memory.
        """
        pass

    def add(self,*,**kwargs):
        r"""Add transition(s) into replay buffer.

        Multple sets of transitions can be added simultaneously.

        Parameters
        ----------
        **kwargs : array like or float or int
            Transitions to be stored.

        Returns
        -------
        : int or None
            The first index of stored position. If all transitions are stored
            into NstepBuffer and no transtions are stored into the main buffer,
            None is returned.

        Raises
        ------
        KeyError
            If any values defined at constructor are missing.

        Warnings
        --------
        All values must be passed by key-value style (keyword arguments).
        It is user responsibility that all the values have the same step-size.
        """
        if self.use_nstep:
            kwargs = self.nstep.add(**kwargs)
            if kwargs is None:
                return

        cdef size_t N = self.size_check.step_size(kwargs)

        cdef size_t index = self.index.fetch_add(N)
        cdef size_t end = index + N
        cdef size_t remain = 0
        cdef add_idx = np.arange(index,end)
        cdef size_t key_min = 0

        if end > self.buffer_size:
            remain = end - self.buffer_size
            add_idx[add_idx >= self.buffer_size] -= self.buffer_size

        if self.compress_any and (remain or
                                  self.get_stored_size() == self.buffer_size):
            key_min = remain or end
            for key in range(key_min,
                             min(key_min + self.cache_size, self.buffer_size)):
                self.add_cache_i(key, index)

        for name, b in self.buffer.items():
            b[add_idx] = np.reshape(np.array(kwargs[name],copy=False,ndmin=2),
                                    self.env_dict[name]["add_shape"])

        if self.has_next_of:
            for name in self.next_of:
                self.next_[name][...]=np.reshape(np.array(kwargs[f"next_{name}"],
                                                          copy=False,
                                                          ndmin=2),
                                                 self.env_dict[name]["add_shape"])[-1]

        if (self.cache is not None) and (index in self.cache):
            del self.cache[index]

        self.episode_len += N
        return index

    def get_all_transitions(self,shuffle: bool=False):
        r"""
        Get all transitions stored in replay buffer.

        Parameters
        ----------
        shuffle : bool, optional
            When True, transitions are shuffled. The default value is False.

        Returns
        -------
        transitions : dict of numpy.ndarray
            All transitions stored in this replay buffer.
        """
        idx = np.arange(self.get_stored_size())

        if shuffle:
            np.random.shuffle(idx)

        return self._encode_sample(idx)

    def save_transitions(self, file, *, safe=True):
        r"""
        Save transitions to file

        Parameters
        ----------
        file : str or file-like object
            File to write data
        safe : bool, optional
            If `False`, we try more aggressive compression
            which might encounter future incompatibility
        """
        FORMAT_VERSION = 1
        if (safe or not (self.compress_any or self.has_next_of)):
            data = {"safe": True,
                    "version": FORMAT_VERSION,
                    "data": self.get_all_transitions(),
                    "Nstep": self.is_Nstep(),
                    "cache": None}
        else:
            data = {"safe": False,
                    "version": FORMAT_VERSION,
                    "data": self.buffer,
                    "Nstep": self.is_Nstep(),
                    "cache": self.cache}
        np.savez_compressed(file, **data)

    def _load_transitions_v1(self, data):
        d = unwrap(data["data"])

        if not data["safe"]:
            c = unwrap(data["cache"])
            N = next(d.values()).shape[0]

            _buffer = self.buffer
            _cache = self.cache

            self.buffer = d
            self.cache = c

            d = self._encode_sample([i for i in range(N)])

            self.buffer = _buffer
            self.cache = _cache

        if data["Nstep"]:
            self.use_nstep = False
            self.add(**d)
            self.use_nstep = True
        else:
            self.add(**d)

    def load_transitions(self, file):
        r"""
        Load transitions from file

        Parameters
        ----------
        file : str or file-like object
            File to read data

        Raises
        ------
        ValueError : When file format is wrong.

        Warnings
        --------
        In order to avoid security vulnerability,
        you MUST NOT load untrusted file, since this method is
        based on `pickle` through `joblib.load`.
        """
        with np.load(file, allow_pickle=True) as data:
            version = data["version"]
            N = data["Nstep"]

            if (N and not self.is_Nstep()) or (not N and self.is_Nstep()):
                raise ValueError(f"Stored data and Buffer mismatch for Nstep")

            if version == 1:
                self._load_transitions_v1(data)
            else:
                raise ValueError(f"Unknown Format Version: {version}")

    def _encode_sample(self,idx):
        cdef sample = {}
        cdef next_idx
        cdef cache_idx
        cdef bool use_cache

        idx = np.array(idx,copy=False,ndmin=1)
        for name, b in self.buffer.items():
            sample[name] = b[idx]

        if self.has_next_of:
            next_idx = idx + 1
            next_idx[next_idx == self.get_buffer_size()] = 0
            cache_idx = (next_idx == self.get_next_index())
            use_cache = cache_idx.any()

            for name in self.next_of:
                sample[f"next_{name}"] = self.buffer[name][next_idx]
                if use_cache:
                    # Cache for the latest "next_***" stored at `self.next_`
                    sample[f"next_{name}"][cache_idx] = self.next_[name]

        cdef size_t i,_i
        cdef size_t N = idx.shape[0]
        if self.cache is not None:
            # Cache for episode ends stored at `self.cache`
            for _i in range(N):
                i = idx[_i]
                if i in self.cache:
                    if self.has_next_of:
                        for name in self.next_of:
                            sample[f"next_{name}"][_i] = self.cache[i][f"next_{name}"]
                    if self.compress_any:
                        for name in self.stack_compress:
                            sample[name][_i] = self.cache[i][name]

        return sample

    def sample(self,batch_size):
        r"""Sample the stored transitions randomly with speciped size

        Parameters
        ----------
        batch_size : int
            sampled batch size

        Returns
        -------
        sample : dict of ndarray
            Batch size of sampled transitions, which might contains
            the same transition multiple times.
        """
        cdef idx = np.random.randint(0,self.get_stored_size(),batch_size)
        return self._encode_sample(idx)

    cpdef void clear(self) except *:
        r"""Clear replay buffer.

        Set `index` and `stored_size` to 0.

        Example
        -------
        >>> rb = ReplayBuffer(5,{"done",{}})
        >>> rb.add(1)
        >>> rb.get_stored_size()
        1
        >>> rb.get_next_index()
        1
        >>> rb.clear()
        >>> rb.get_stored_size()
        0
        >>> rb.get_next_index()
        0
        """
        self.index.clear()
        self.episode_len = 0

        self.cache = {} if (self.has_next_of or self.compress_any) else None

        if self.use_nstep:
            self.nstep.clear()

    cpdef size_t get_stored_size(self):
        r"""Get stored size

        Returns
        -------
        size_t
            stored size
        """
        return self.index.get_stored_size()

    cpdef size_t get_buffer_size(self):
        r"""Get buffer size

        Returns
        -------
        size_t
            buffer size
        """
        return self.buffer_size

    cpdef size_t get_next_index(self):
        r"""Get the next index to store

        Returns
        -------
        size_t
            the next index to store
        """
        return self.index.get_next_index()

    cdef void add_cache(self):
        r"""Add last items into cache

        The last items for "next_of" and "stack_compress" optimization
        are moved to cache area.

        If `self.cache is None`, do nothing.
        If `self.stored_size == 0`, do nothing.
        """

        # If no cache configuration, do nothing
        if self.cache is None:
            return

        # If nothing are stored, do nothing
        if self.get_stored_size() == 0:
            return

        cdef size_t key_end = (self.get_next_index() or self.buffer_size)
        # Next index (without wraparounding): key_end in [1,...,self.buffer_size]

        cdef size_t key_min = 0
        cdef size_t max_cache = min(self.cache_size,self.episode_len)
        if key_end > max_cache:
            key_min = key_end - max_cache

        cdef size_t key = 0
        cdef size_t next_key = 0
        for key in range(key_min, key_end): # key_end is excluded
            self.add_cache_i(key, key_end)

    cdef void add_cache_i(self, size_t key, size_t key_end):
        # If key is already cached, don't do anything
        if key in self.cache:
            return

        cdef size_t next_key = key + 1
        cdef cache_key = {}

        if self.has_next_of:
            if next_key == key_end:
                for name, value in self.next_.items():
                    cache_key[f"next_{name}"] = value.copy()
            else:
                for name in self.next_.keys():
                    cache_key[f"next_{name}"] = self.buffer[name][next_key].copy()

        if self.compress_any:
            for name in self.stack_compress:
                cache_key[name] = self.buffer[name][key].copy()

        self.cache[key] = cache_key



    cpdef void on_episode_end(self) except *:
        r"""Call on episode end

        Finalize the current episode by moving remaining Nstep buffer transitions,
        evacuating overlapped data for memory compression features, and resetting
        episode length.

        Notes
        -----
        Calling this function at episode end is the user responsibility,
        since episode exploration can be terminated at certain length
        even though any `done` flags from environment is not set.
        """
        if self.use_nstep:
            self.use_nstep = False
            self.add(**self.nstep.on_episode_end())
            self.use_nstep = True

        self.add_cache()

        self.episode_len = 0

    cpdef size_t get_current_episode_len(self):
        r"""Get current episode length

        Returns
        -------
        episode_len : size_t
        """
        return self.episode_len

    cpdef bool is_Nstep(self):
        r"""Get whether use Nstep or not

        Returns
        -------
        use_nstep : bool
        """
        return self.use_nstep

@cython.embedsignature(True)
cdef class PrioritizedReplayBuffer(ReplayBuffer):
    r"""Prioritized replay buffer class to store transitions with priorities.

    In this class, these transitions are sampled with corresponding to priorities.
    """
    cdef VectorFloat weights
    cdef VectorSize_t indexes
    cdef float alpha
    cdef CppPrioritizedSampler[float]* per
    cdef NstepBuffer priorities_nstep
    cdef bool check_for_update
    cdef bool [:] unchange_since_sample
    cdef vector[size_t] idx_vec
    cdef vector[float] ps_vec

    def __cinit__(self,size,env_dict=None,*,alpha=0.6,Nstep=None,eps=1e-4,
                  check_for_update=False,**kwrags):
        self.alpha = alpha
        self.per = new CppPrioritizedSampler[float](size,alpha)
        self.per.set_eps(eps)
        self.weights = VectorFloat()
        self.indexes = VectorSize_t()

        if self.use_nstep:
            self.priorities_nstep = NstepBuffer({"priorities": {"dtype": np.single},
                                                 "done": {}},
                                                {"size": Nstep["size"]})

        self.check_for_update = check_for_update
        if self.check_for_update:
            self.unchange_since_sample = np.ones(np.array(size,
                                                          copy=False,
                                                          dtype='int'),
                                                 dtype='bool')

        self.idx_vec = vector[size_t]()
        self.ps_vec = vector[float]()

    def __init__(self,size,env_dict=None,*,alpha=0.6,Nstep=None,eps=1e-4,
                 check_for_update=False,**kwargs):
        r"""Initialize PrioritizedReplayBuffer

        Parameters
        ----------
        size : int
            buffer size
        env_dict : dict of dict, optional
            dictionary specifying environments. The keies of env_dict become
            environment names. The values of env_dict, which are also dict,
            defines "shape" (default 1) and "dtypes" (fallback to `default_dtype`)
        alpha : float, optional
            :math:`\alpha` the exponent of the priorities in stored whose
            default value is 0.6
        eps : float, optional
            :math:`\epsilon` small positive constant to ensure error-less state
            will be sampled, whose default value is 1e-4.
        check_for_update : bool
            If the value is `True` (default value is `False`),
            this buffer traces updated indices after the last calling of
            `sample()` method to avoid mis-updating priorities of already
            overwritten values. This feature is designed for multiprocess learning.

        See Also
        --------
        ReplayBuffer : Any optional parameters at ReplayBuffer are valid, too.


        Notes
        -----
        The minimum and summation over certain ranges of pre-calculated priorities
        :math:`(p_{i} + \epsilon )^{ \alpha }` are stored with segment tree, which
        enable fast sampling.
        """
        pass

    def add(self,*,priorities = None,**kwargs):
        r"""Add transition(s) into replay buffer.

        Multple sets of transitions can be added simultaneously.

        Parameters
        ----------
        priorities : array like or float, optional
            Priorities of each environment. When no priorities are passed,
            the maximum priorities until then are used.
        **kwargs : array like or float or int
            Transitions to be stored.

        Returns
        -------
        : int or None
            The first index of stored position. If all transitions are stored
            into NstepBuffer and no transtions are stored into the main buffer,
            None is returned.

        Raises
        ------
        KeyError
            If any values defined at constructor are missing.

        Warnings
        --------
        All values must be passed by key-value style (keyword arguments).
        It is user responsibility that all the values have the same step-size.
        """
        cdef size_t N = self.size_check.step_size(kwargs)
        if priorities is not None:
            priorities = np.ravel(np.array(priorities,copy=False,
                                           ndmin=1,dtype=np.single))
            if N != priorities.shape[0]:
                raise ValueError("`priorities` shape is imcompatible")

        if self.use_nstep:
            if priorities is None:
                priorities = np.full((N),self.get_max_priority(),dtype=np.single)

            priorities = self.priorities_nstep.add(priorities=priorities,
                                                   done=np.array(kwargs["done"],
                                                                 copy=True))
            if priorities is not None:
                priorities = np.ravel(priorities["priorities"])
                N = priorities.shape[0]

        cdef maybe_index = super().add(**kwargs)
        if maybe_index is None:
            return None

        cdef size_t index = maybe_index
        cdef const float [:] ps

        if priorities is not None:
            ps = np.ravel(np.array(priorities,copy=False,ndmin=1,dtype=np.single))
            self.per.set_priorities(index,&ps[0],N,self.get_buffer_size())
        else:
            self.per.set_priorities(index,N,self.get_buffer_size())

        if self.check_for_update:
            if index+N <= self.buffer_size:
                self.unchange_since_sample[index:index+N] = False
            else:
                self.unchange_since_sample[index:] = False
                self.unchange_since_sample[:index+N-self.buffer_size] = False

        return index

    def sample(self,batch_size,beta = 0.4):
        r"""Sample the stored transitions.

        Transisions are sampled depending on correspoinding priorities
        with speciped size

        Parameters
        ----------
        batch_size : int
            Sampled batch size
        beta : float, optional
            The exponent of weight for relaxation of importance
            sampling effect, whose default value is 0.4

        Returns
        -------
        sample : dict of ndarray
            Batch size of samples which also includes 'weights' and 'indexes'

        Notes
        -----
        When 'beta' is 0, weights become uniform. Wen 'beta' is 1, weight becomes
        usual importance sampling.
        The 'weights' are also normalized by the weight for minimum priority
        (:math:`= w_{i}/\max_{j}(w_{j})`), which ensure the weights :math:`\leq` 1.
        """
        self.per.sample(batch_size,beta,
                        self.weights.vec,self.indexes.vec,
                        self.get_stored_size())
        cdef idx = self.indexes.as_numpy()
        samples = self._encode_sample(idx)
        samples['weights'] = self.weights.as_numpy()
        samples['indexes'] = idx

        if self.check_for_update:
            self.unchange_since_sample[:] = True

        return samples

    def update_priorities(self,indexes,priorities):
        r"""Update priorities

        Update priorities specified with indicies. If this
        PrioritizedReplayBuffer is constructed with
        `check_for_update=True`, then ignore indices which updated
        values after the last calling of `sample()` method.

        Parameters
        ----------
        indexes : array_like
            indexes to update priorities
        priorities : array_like
            priorities to update

        Raises
        ------
        TypeError: When `indexes` or `priorities` are `None`
        """

        if priorities is None:
            raise TypeError("`properties` must not be `None`")

        cdef const size_t [:] idx = Csize(indexes)
        cdef const float [:] ps = Cfloat(priorities)

        if not self.check_for_update:
            self.per.update_priorities(&idx[0],&ps[0],idx.shape[0])
            return None

        self.idx_vec.clear()
        self.idx_vec.reserve(idx.shape[0])

        self.ps_vec.clear()
        self.ps_vec.reserve(ps.shape[0])

        if self.check_for_update:
            for _i in range(idx.shape[0]):
                if self.unchange_since_sample[idx[_i]]:
                    self.idx_vec.push_back(idx[_i])
                    self.ps_vec.push_back(ps[_i])

        cdef N = self.idx_vec.size()
        if N > 0:
            self.per.update_priorities(self.idx_vec.data(),self.ps_vec.data(),N)

    cpdef void clear(self) except *:
        r"""Clear replay buffer
        """
        super(PrioritizedReplayBuffer,self).clear()
        clear(self.per)
        if self.use_nstep:
            self.priorities_nstep.clear()

    cpdef float get_max_priority(self):
        r"""Get the max priority of stored priorities

        Returns
        -------
        max_priority : float
            the max priority of stored priorities
        """
        return self.per.get_max_priority()

    cpdef void on_episode_end(self) except *:
        r"""Call on episode end

        Finalize the current episode by moving remaining Nstep buffer transitions,
        evacuating overlapped data for memory compression features, and resetting
        episode length.

        Notes
        -----
        Calling this function at episode end is the user responsibility,
        since episode exploration can be terminated at certain length
        even though any `done` flags from environment is not set.
        """
        if self.use_nstep:
            self.use_nstep = False
            self.add(**self.nstep.on_episode_end(),
                     priorities=self.priorities_nstep.on_episode_end()["priorities"])
            self.use_nstep = True

        self.add_cache()

        self.episode_len = 0


@cython.embedsignature(True)
cdef class MPReplayBuffer:
    r"""Multi-process support Replay Buffer class to store transitions and to sample them randomly.

    This class works on multi-process without manual locking of entire buffer.

    The transition can contain anything compatible with numpy data
    type. User can specify by `env_dict` parameters at constructor
    freely.

    The possible standard transition contains observation (`obs`), action (`act`),
    reward (`rew`), the next observation (`next_obs`), and done (`done`).

    >>> env_dict = {"obs": {"shape": (4,4)},
                    "act": {"shape": 3, "dtype": np.int16},
                    "rew": {},
                    "next_obs": {"shape": (4,4)},
                    "done": {}}

    In this class, sampling is random sampling and the same transition
    can be chosen multiple times.

    Notes
    -----
    This class assumes single learner (`sample`) and multiple explorers (`add`)
    like Ape-X
    """
    cdef buffer
    cdef size_t buffer_size
    cdef env_dict
    cdef ProcessSafeRingBufferIndex index
    cdef default_dtype
    cdef StepChecker size_check
    cdef explorer_ready
    cdef explorer_count
    cdef learner_ready

    def __init__(self,size,env_dict=None,*,default_dtype=None,logger=None,**kwargs):
        r"""Initialize ReplayBuffer

        Parameters
        ----------
        size : int
            buffer size
        env_dict : dict of dict, optional
            dictionary specifying environments. The keies of env_dict become
            environment names. The values of env_dict, which are also dict,
            defines "shape" (default 1) and "dtypes" (fallback to `default_dtype`)
        default_dtype : numpy.dtype, optional
            fallback dtype for not specified in `env_dict`. default is numpy.single
        """
        self.env_dict = env_dict.copy() if env_dict else {}
        cdef special_keys = []

        self.buffer_size = size
        self.index = ProcessSafeRingBufferIndex(self.buffer_size)

        self.default_dtype = default_dtype or np.single

        # side effect: Add "add_shape" key into self.env_dict
        self.buffer = dict2buffer(self.buffer_size,self.env_dict,
                                  default_dtype = self.default_dtype,
                                  shared = True)

        self.size_check = StepChecker(self.env_dict,special_keys)

        self.learner_ready = Event()
        self.learner_ready.clear()
        self.explorer_ready = Event()
        self.explorer_ready.set()

        self.explorer_count = Value(ctypes.c_size_t,0)

    cdef void _lock_explorer(self) except *:
        self.explorer_ready.wait() # Wait permission
        self.learner_ready.clear()  # Block learner
        with self.explorer_count.get_lock():
            self.explorer_count.value += 1

    cdef void _unlock_explorer(self) except *:
        with self.explorer_count.get_lock():
            self.explorer_count.value -= 1
        if self.explorer_count.value == 0:
            self.learner_ready.set()

    cdef void _lock_learner(self) except *:
        self.explorer_ready.clear() # New explorer cannot enter into critical section
        self.learner_ready.wait() # Wait until all explorer exit from critical section

    cdef void _unlock_learner(self) except *:
        self.explorer_ready.set() # Allow workers to enter into critical section

    def add(self,*,**kwargs):
        r"""Add transition(s) into replay buffer.

        Multple sets of transitions can be added simultaneously. This method
        can be called from multiple explorer processes without manual lock.

        Parameters
        ----------
        **kwargs : array like or float or int
            Transitions to be stored.

        Returns
        -------
        : int or None
            The first index of stored position. If all transitions are stored
            into NstepBuffer and no transtions are stored into the main buffer,
            None is returned.

        Raises
        ------
        KeyError
            If any values defined at constructor are missing.

        Warnings
        --------
        All values must be passed by key-value style (keyword arguments).
        It is user responsibility that all the values have the same step-size.
        """
        cdef size_t N = self.size_check.step_size(kwargs)

        cdef size_t index = self.index.fetch_add(N)
        cdef size_t end = index + N
        cdef add_idx = np.arange(index,end)

        if end > self.buffer_size:
            add_idx[add_idx >= self.buffer_size] -= self.buffer_size


        self._lock_explorer()

        for name, b in self.buffer.items():
            b[add_idx] = np.reshape(np.array(kwargs[name],copy=False,ndmin=2),
                                    self.env_dict[name]["add_shape"])

        self._unlock_explorer()
        return index

    def get_all_transitions(self,shuffle: bool=False):
        r"""
        Get all transitions stored in replay buffer.

        Parameters
        ----------
        shuffle : bool, optional
            When True, transitions are shuffled. The default value is False.

        Returns
        -------
        transitions : dict of numpy.ndarray
            All transitions stored in this replay buffer.
        """
        idx = np.arange(self.get_stored_size())

        if shuffle:
            np.random.shuffle(idx)

        self._lock_learner()
        ret = self._encode_sample(idx)
        self._unlock_learner()

        return ret

    def _encode_sample(self,idx):
        cdef sample = {}

        idx = np.array(idx,copy=False,ndmin=1)

        for name, b in self.buffer.items():
            sample[name] = b[idx]

        return sample

    def sample(self,batch_size):
        r"""Sample the stored transitions randomly with speciped size

        This method can be called from a single learner process.

        Parameters
        ----------
        batch_size : int
            sampled batch size

        Returns
        -------
        sample : dict of ndarray
            Batch size of sampled transitions, which might contains
            the same transition multiple times.
        """
        cdef idx = np.random.randint(0,self.get_stored_size(),batch_size)

        self._lock_learner()
        ret =  self._encode_sample(idx)
        self._unlock_learner()

        return ret

    cpdef void clear(self) except *:
        r"""Clear replay buffer.

        Set `index` and `stored_size` to 0.

        Example
        -------
        >>> rb = ReplayBuffer(5,{"done",{}})
        >>> rb.add(1)
        >>> rb.get_stored_size()
        1
        >>> rb.get_next_index()
        1
        >>> rb.clear()
        >>> rb.get_stored_size()
        0
        >>> rb.get_next_index()
        0
        """
        self.index.clear()

    cpdef size_t get_stored_size(self):
        r"""Get stored size

        Returns
        -------
        size_t
            stored size
        """
        return self.index.get_stored_size()

    cpdef size_t get_buffer_size(self):
        r"""Get buffer size

        Returns
        -------
        size_t
            buffer size
        """
        return self.buffer_size

    cpdef size_t get_next_index(self):
        r"""Get the next index to store

        Returns
        -------
        size_t
            the next index to store
        """
        return self.index.get_next_index()

    cpdef void on_episode_end(self) except *:
        r"""Call on episode end

        Finalize the current episode by moving remaining Nstep buffer transitions,
        evacuating overlapped data for memory compression features, and resetting
        episode length.

        Notes
        -----
        Calling this function at episode end is the user responsibility,
        since episode exploration can be terminated at certain length
        even though any `done` flags from environment is not set.
        """
        pass

    cpdef bool is_Nstep(self):
        r"""Get whether use Nstep or not

        Returns
        -------
        use_nstep : bool
        """
        return False


cdef class ThreadSafePrioritizedSampler:
    cdef size_t size
    cdef float alpha
    cdef float eps
    cdef max_p
    cdef sum
    cdef sum_a#nychanged
    cdef min
    cdef min_a#nychanged
    cdef CppThreadSafePrioritizedSampler[float]* per

    def __init__(self,size,alpha,eps,max_p=None,
                 sum=None,sum_a=None,
                 min=None,min_a=None):
        self.size = size
        self.alpha = alpha
        self.eps = eps

        self.max_p = max_p or RawArray(ctypes.c_float,1)
        cdef float [:] view_max_p = self.max_p

        cdef size_t pow2size = 1
        while pow2size < size:
            pow2size *= 2

        self.sum   = sum   or RawArray(ctypes.c_float,2*pow2size-1)
        self.sum_a = sum_a or RawArray(ctypes.c_bool ,1)
        self.min   = min   or RawArray(ctypes.c_float,2*pow2size-1)
        self.min_a = min_a or RawArray(ctypes.c_bool ,1)

        cdef float [:] view_sum   = self.sum
        cdef bool  [:] view_sum_a = self.sum_a
        cdef float [:] view_min   = self.min
        cdef bool  [:] view_min_a = self.min_a

        cdef bool init = ((max_p is None) and
                          (sum   is None) and
                          (sum_a is None) and
                          (min   is None) and
                          (min_a is None))

        self.per = new CppThreadSafePrioritizedSampler[float](size,alpha,
                                                              &view_max_p[0],
                                                              &view_sum[0],
                                                              &view_sum_a[0],
                                                              &view_min[0],
                                                              &view_min_a[0],
                                                              init,
                                                              eps)

    cdef CppThreadSafePrioritizedSampler[float]* ptr(self):
        return self.per

    def __reduce__(self):
        return (ThreadSafePrioritizedSampler,
                (self.size,self.alpha,self.eps,self.max_p,
                 self.sum,self.sum_a,
                 self.min,self.min_a))


@cython.embedsignature(True)
cdef class MPPrioritizedReplayBuffer(MPReplayBuffer):
    r"""Multi-process support Prioritized Replay Buffer class to store transitions with priorities.

    This class can work on multi-process without manual lock.

    In this class, these transitions are sampled with corresponding to priorities.

    Notes
    -----
    This class assumes single learner (`sample`, `update_priorities`) and
    multiple explorers (`add`).
    """
    cdef VectorFloat weights
    cdef VectorSize_t indexes
    cdef ThreadSafePrioritizedSampler per
    cdef unchange_since_sample
    cdef helper
    cdef terminate
    cdef explorer_per_count
    cdef learner_per_ready
    cdef explorer_per_ready
    cdef vector[size_t] idx_vec
    cdef vector[float] ps_vec

    def __init__(self,size,env_dict=None,*,alpha=0.6,eps=1e-4,**kwargs):
        r"""Initialize PrioritizedReplayBuffer

        Parameters
        ----------
        size : int
            buffer size
        env_dict : dict of dict, optional
            dictionary specifying environments. The keies of env_dict become
            environment names. The values of env_dict, which are also dict,
            defines "shape" (default 1) and "dtypes" (fallback to `default_dtype`)
        alpha : float, optional
            :math:`\alpha` the exponent of the priorities in stored whose
            default value is 0.6
        eps : float, optional
            :math:`\epsilon` small positive constant to ensure error-less state
            will be sampled, whose default value is 1e-4.

        See Also
        --------
        ReplayBuffer : Any optional parameters at ReplayBuffer are valid, too.


        Notes
        -----
        The minimum and summation over certain ranges of pre-calculated priorities
        :math:`(p_{i} + \epsilon )^{ \alpha }` are stored with segment tree, which
        enable fast sampling.
        """
        super().__init__(size,env_dict,**kwargs)

        self.per = ThreadSafePrioritizedSampler(size,alpha,eps)

        self.weights = VectorFloat()
        self.indexes = VectorSize_t()

        shm = RawArray(np.ctypeslib.as_ctypes_type(np.bool_),
                       int(np.array(size,copy=False,dtype='int').prod()))
        self.unchange_since_sample = np.ctypeslib.as_array(shm)
        self.unchange_since_sample[:] = True

        self.helper = None
        self.terminate = Value(ctypes.c_bool)
        self.terminate.value = False

        self.learner_per_ready = Event()
        self.learner_per_ready.clear()
        self.explorer_per_ready = Event()
        self.explorer_per_ready.set()
        self.explorer_per_count = Value(ctypes.c_size_t,0)

        self.idx_vec = vector[size_t]()
        self.ps_vec = vector[float]()

    cdef void _lock_explorer_per(self) except *:
        self.explorer_per_ready.wait() # Wait permission
        self.learner_per_ready.clear()  # Block learner
        with self.explorer_per_count.get_lock():
            self.explorer_per_count.value += 1

    cdef void _unlock_explorer_per(self) except *:
        with self.explorer_per_count.get_lock():
            self.explorer_per_count.value -= 1
        if self.explorer_per_count.value == 0:
            self.learner_per_ready.set()

    cdef void _lock_learner_per(self) except *:
        self.explorer_per_ready.clear()
        self.learner_per_ready.wait()

    cdef void _unlock_learner_per(self) except *:
        self.explorer_per_ready.set()

    cdef void _lock_learner_unlock_learner_per(self) except *:
        self.explorer_ready.clear()
        self.explorer_per_ready.set()
        self.learner_ready.wait()

    def add(self,*,priorities = None,**kwargs):
        r"""Add transition(s) into replay buffer.

        Multple sets of transitions can be added simultaneously. This method can be
        called from multiple explorer processes without manual lock.

        Parameters
        ----------
        priorities : array like or float, optional
            Priorities of each environment. When no priorities are passed,
            the maximum priorities until then are used.
        **kwargs : array like or float or int
            Transitions to be stored.

        Returns
        -------
        : int or None
            The first index of stored position. If all transitions are stored
            into NstepBuffer and no transtions are stored into the main buffer,
            None is returned.

        Raises
        ------
        KeyError
            If any values defined at constructor are missing.

        Warnings
        --------
        All values must be passed by key-value style (keyword arguments).
        It is user responsibility that all the values have the same step-size.
        """
        cdef size_t N = self.size_check.step_size(kwargs)
        cdef const float [:] ps

        if priorities is not None:
            priorities = np.ravel(np.array(priorities,copy=False,
                                           ndmin=1,dtype=np.single))
            if N != priorities.shape[0]:
                raise ValueError("`priorities` shape is imcompatible")

        cdef size_t index = self.index.fetch_add(N)
        cdef size_t end = index + N
        cdef add_idx = np.arange(index,end)

        if end > self.buffer_size:
            add_idx[add_idx >= self.buffer_size] -= self.buffer_size


        self._lock_explorer_per()

        if priorities is not None:
            ps = np.ravel(np.array(priorities,copy=False,ndmin=1,dtype=np.single))
            self.per.ptr().set_priorities(index,&ps[0],N,self.get_buffer_size())
        else:
            self.per.ptr().set_priorities(index,N,self.get_buffer_size())

        if index+N <= self.buffer_size:
            self.unchange_since_sample[index:index+N] = False
        else:
            self.unchange_since_sample[index:] = False
            self.unchange_since_sample[:index+N-self.buffer_size] = False

        self._lock_explorer()
        self._unlock_explorer_per()

        for name, b in self.buffer.items():
            b[add_idx] = np.reshape(np.array(kwargs[name],copy=False,ndmin=2),
                                    self.env_dict[name]["add_shape"])

        self._unlock_explorer()
        return index

    def sample(self,batch_size,beta = 0.4):
        r"""Sample the stored transitions.

        Transisions are sampled depending on correspoinding priorities
        with speciped size. This method can be called from single learner process.

        Parameters
        ----------
        batch_size : int
            Sampled batch size
        beta : float, optional
            The exponent of weight for relaxation of importance
            sampling effect, whose default value is 0.4

        Returns
        -------
        sample : dict of ndarray
            Batch size of samples which also includes 'weights' and 'indexes'

        Notes
        -----
        When 'beta' is 0, weights become uniform. Wen 'beta' is 1, weight becomes
        usual importance sampling.
        The 'weights' are also normalized by the weight for minimum priority
        (:math:`= w_{i}/\max_{j}(w_{j})`), which ensure the weights :math:`\leq` 1.
        """
        self._lock_learner_per()
        self.per.ptr().sample(batch_size,beta,
                              self.weights.vec,self.indexes.vec,
                              self.get_stored_size())
        cdef idx = self.indexes.as_numpy()

        self._lock_learner_unlock_learner_per()

        samples = self._encode_sample(idx)
        self.unchange_since_sample[:] = True
        self._unlock_learner()

        samples['weights'] = self.weights.as_numpy()
        samples['indexes'] = idx

        return samples

    def update_priorities(self,indexes,priorities):
        r"""Update priorities

        Update priorities specified with indicies. Ignores indices
        which updated values after the last calling of `sample()`
        method. This method can be called from single learner process.

        Parameters
        ----------
        indexes : array_like
            indexes to update priorities
        priorities : array_like
            priorities to update

        Raises
        ------
        TypeError: When `indexes` or `priorities` are `None`
        """

        if priorities is None:
            raise TypeError("`properties` must not be `None`")

        cdef const size_t [:] idx = Csize(indexes)
        cdef const float [:] ps = Cfloat(priorities)

        self.idx_vec.clear()
        self.idx_vec.reserve(idx.shape[0])

        self.ps_vec.clear()
        self.ps_vec.reserve(ps.shape[0])

        self._lock_learner_per()
        cdef size_t stored_size = self.get_stored_size()
        for _i in range(idx.shape[0]):
            if idx[_i] < stored_size and self.unchange_since_sample[idx[_i]]:
                self.idx_vec.push_back(idx[_i])
                self.ps_vec.push_back(ps[_i])

        cdef N = self.idx_vec.size()
        if N > 0:
            self.per.ptr().update_priorities(self.idx_vec.data(),self.ps_vec.data(),N)
        self._unlock_learner_per()

    cpdef void clear(self) except *:
        r"""Clear replay buffer
        """
        super(MPPrioritizedReplayBuffer,self).clear()
        clear(self.per.ptr())

    cpdef float get_max_priority(self):
        r"""Get the max priority of stored priorities

        Returns
        -------
        max_priority : float
            the max priority of stored priorities
        """
        return self.per.ptr().get_max_priority()

    cpdef void on_episode_end(self) except *:
        r"""Call on episode end

        Notes
        -----
        Calling this function at episode end is the user responsibility,
        since episode exploration can be terminated at certain length
        even though any `done` flags from environment is not set.
        """
        pass


@cython.embedsignature(True)
def create_buffer(size,env_dict=None,*,prioritized = False,**kwargs):
    r"""Create specified version of replay buffer

    Parameters
    ----------
    size : int
        buffer size
    env_dict : dict of dict, optional
        dictionary specifying environments. The keies of env_dict become
        environment names. The values of env_dict, which are also dict,
        defines "shape" (default 1) and "dtypes" (fallback to `default_dtype`)
    prioritized : bool, optional
        create prioritized version replay buffer, default = False

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

    buffer_name = f"{per}ReplayBuffer"

    cls={"ReplayBuffer": ReplayBuffer,
         "PrioritizedReplayBuffer": PrioritizedReplayBuffer}

    buffer = cls.get(f"{buffer_name}",None)

    if buffer:
        return buffer(size,env_dict,**kwargs)

    raise NotImplementedError(f"{buffer_name} is not Implemented")


@cython.embedsignature(True)
def train(buffer: ReplayBuffer,
          env,
          get_action: Callable,
          update_policy: Callable,*,
          max_steps: int=int(1e6),
          max_episodes: Optional[int] = None,
          batch_size: int = 64,
          n_warmups: int = 0,
          after_step: Optional[Callable] = None,
          done_check: Optional[Callable] = None,
          obs_update: Optional[Callable] = None,
          rew_sum: Optional[Callable[[float, Any], float]] = None,
          episode_callback: Optional[Callable[[int,int,float],Any]] = None,
          logger = None):
    r"""
    Train RL policy (model)

    Parameters
    ----------
    buffer: ReplayBuffer
        Buffer to be used for training
    env: gym.Enviroment compatible
        Environment to learn
    get_action: Callable
        Callable taking `obs` and returning `action`
    update_policy: Callable
        Callable taking `sample`, `step`, and `episode`, updating policy,
        and returning |TD|.
    max_steps: int (optional)
        Maximum steps to learn. The default value is `1000000`
    max_episodes: int (optional)
        Maximum episodes to learn. The defaul value is `None`
    n_warmups: int (optional)
        Warmup steps before sampling. The default value is `0` (No warmup)
    after_step: Callable (optional)
        Callable converting from `obs`, returns of `env.step(action)`,
        `step`, and `episode` to `dict` of a transition for `ReplayBuffer.add`.
        This function can also be used for step summary callback.
    done_check: Callable (optional)
        Callable checking done
    obs_update: Callable (optional)
        Callable updating obs
    rew_sum: Callable[[float, Dict], float] (optional)
        Callable summarizing episode reward
    episode_callback: Callable[[int, int, float], Any] (optional)
        Callable for episode summarization
    logger: logging.Logger (optional)
        Custom Logger

    Raises
    ------
    ValueError:
       When `max_step` is larger than `size_t` limit

    Warnings
    --------
    `cpprb.train` is still beta release. API can be changed.
    """
    warnings.warn("`cpprb.train` is still beta release. API can be changed.")

    logger = logger or default_logger()

    cdef size_t size_t_limit = -1
    if max_steps >= int(size_t_limit):
        raise ValueError(f"max_steps ({max_steps}) is too big. " +
                         f"max_steps < {size_t_limit}")

    cdef bool use_per = isinstance(buffer,PrioritizedReplayBuffer)
    cdef bool has_after_step = after_step
    cdef bool has_check = done_check
    cdef bool has_obs_update = obs_update
    cdef bool has_rew_sum = rew_sum
    cdef bool has_episode_callback = episode_callback

    cdef size_t _max_steps = max(max_steps,0)
    cdef size_t _max_episodes = min(max(max_episodes or size_t_limit, 0),size_t_limit)
    cdef size_t _n_warmup = min(max(0,n_warmups),size_t_limit)

    cdef size_t step = 0
    cdef size_t episode = 0
    cdef size_t episode_step = 0
    cdef float episode_reward = 0.0
    cdef bool is_warmup = True

    obs = env.reset()
    cdef double episode_start_time = time.perf_counter()
    cdef double episode_end_time = 0.0
    for step in range(_max_steps):
        is_warmup = (step < _n_warmup)

        # Get action
        action = get_action(obs,step,episode,is_warmup)

        # Step environment
        if has_after_step:
            transition = after_step(obs,action,env.step(action),step,episode)
        else:
            next_obs, reward, done, _ = env.step(action)
            transition = {"obs": obs,
                          "act": action,
                          "rew": reward,
                          "next_obs": next_obs,
                          "done": done}

        # Add to buffer
        buffer.add(**transition)

        # For Nstep, ReplayBuffer can be empty after `add(**transition)` method
        if (buffer.get_stored_size() > 0) and (not is_warmup):
            # Sample
            sample = buffer.sample(batch_size)
            absTD = update_policy(sample,step,episode)

            if use_per:
                buffer.update_priorities(sample["indexes"],absTD)

        # Summarize reward
        episode_reward = (rew_sum(episode_reward,transition) if has_rew_sum
                          else transition["rew"])

        # Prepare the next step
        if done_check(transition) if has_check else transition["done"]:
            episode_end_time = time.perf_counter()

            # step/episode_step are index.
            # Total Steps/Episode Steps are counts.
            SPS = (episode_step+1) / max(episode_end_time-episode_start_time,1e-9)
            logger.info(f"{episode: 6}th Episode: " +
                        f"{episode_step+1: 5} Steps " +
                        f"({step+1: 7} Total Steps), " +
                        f"{episode_reward: =+7.2f} Reward, " +
                        f"{SPS: =+5.2f} Steps/s")

            # Summary
            if has_episode_callback:
                episode_callback(episode,episode_step,episode_reward)

            # Reset
            obs = env.reset()
            buffer.on_episode_end()
            episode_reward = 0.0
            episode_step = 0

            # Update episode count
            episode += 1
            if episode >= _max_episodes:
                break

            episode_start_time = time.perf_counter()
        else:
            obs = obs_update(transition) if has_obs_update else transition["next_obs"]
            episode_step += 1
