"""
Multiprocessing Module (mod:`cpprb.multiprocessing`)
====================================================

This module implements shared memory backends with compatible interface.
"""

import atexit
import contextlib
import ctypes
import sys
from logging import getLogger
from multiprocessing.context import ProcessError
from multiprocessing.managers import State, SyncManager

import numpy as np

logger = getLogger(__name__)

_has_SharedMemory = sys.version_info >= (3, 8)  # noqa: N816
if _has_SharedMemory:
    # `SharedMemory` class is preferable since it can work even after
    # serialization/deserialization by using unique name. This
    # capability allows users to use buffers in Ray (https://ray.io/).
    from multiprocessing.shared_memory import _USE_POSIX, SharedMemory

    def setup_unlink(shm):
        # Work around a resource tracker issues;
        # * Ref: https://bugs.python.org/issue41447
        # * Ref: https://bugs.python.org/issue38119
        if _USE_POSIX:
            from multiprocessing.resource_tracker import unregister

            import _posixshmem

            name = shm._name  # noqa: SLF001
            unregister(name, "shared_memory")

            @atexit.register
            def unlink(*args):  # noqa: ARG001
                with contextlib.suppress(FileNotFoundError):
                    _posixshmem.shm_unlink(name)
                    # Sometimes, shm file has been desctucted
                    # beforehand, we just ignore.

    class SharedMemoryValue:
        """
        Compatible Value for SharedMemory
        """

        def __init__(self, ctype, init=None):
            size = ctypes.sizeof(ctype)
            self.shm = SharedMemory(create=True, size=size)
            setup_unlink(self.shm)

            self.dtype = np.dtype(ctype)
            if self.dtype.itemsize != size:
                raise ValueError(
                    "BUG: data size mismutch. "
                    f"cpprb failed to handle {ctype}. "
                    "Please report to https://github.com/ymd-h/cpprb/discussions"
                )

            self.ndarray = np.ndarray((1,), self.dtype, buffer=self.shm.buf)
            if init is not None:
                self.ndarray[0] = init

        @property
        def value(self):
            return self.ndarray[0]

        @value.setter
        def value(self, v):
            self.ndarray[0] = v

        def __getstate__(self):
            return (self.shm, self.dtype)

        def __setstate__(self, shm_dtype):
            self.shm, self.dtype = shm_dtype
            self.ndarray = np.ndarray((1,), self.dtype, buffer=self.shm.buf)

    class SharedMemoryArray:
        """
        Compatible Array for SharedMemory
        """

        def __init__(self, ctype, len_):
            self.len = len_
            self.dtype = np.dtype(ctype)

            size = ctypes.sizeof(ctype) * self.len
            self.shm = SharedMemory(create=True, size=size)
            setup_unlink(self.shm)

            self.ndarray = np.ndarray((self.len,), self.dtype, buffer=self.shm.buf)

        def __getstate__(self):
            return (self.shm, self.dtype, self.len)

        def __setstate__(self, shm_dtype_len):
            self.shm, self.dtype, self.len = shm_dtype_len
            self.ndarray = np.ndarray((self.len,), self.dtype, buffer=self.shm.buf)

        def __len__(self):
            return len(self.ndarray)

        def __getitem__(self, i):
            return self.ndarray[i]

        def __setitem__(self, i, value):
            self.ndarray[i] = value


class ctypesArray:  # noqa: N801
    """
    Compatible Array for sharedctypes
    """

    def __init__(self, ctx, ctype, len_):
        self.shm = ctx.Array(ctype, len_, lock=False)
        self.ndarray = np.ctypeslib.as_array(self.shm)

    def __getstate__(self):
        return self.shm

    def __setstate__(self, shm):
        self.shm = shm
        self.ndarray = np.ctypeslib.as_array(self.shm)

    def __len__(self):
        return len(self.ndarray)

    def __getitem__(self, i):
        return self.ndarray[i]

    def __setitem__(self, i, value):
        self.ndarray[i] = value


def RawArray(ctx, ctype, len_, backend):  # noqa: N802
    """
    Get RawArray for backend

    Parameters
    ----------
    ctx
        Multiprocessing Context
    ctype
        C type
    len_ : int
        Length
    backend : {"SharedMemory", "sharedctypes"}
        Backend

    Raises
    ------
    ValueError
        If ``backend`` is unknown.
    """
    if isinstance(ctx, SyncManager):
        ctx = ctx._ctx  # noqa: SLF001
    len_ = int(len_)
    if not _has_SharedMemory and backend == "SharedMemory":
        backend = "sharedctypes"
        logger.warning(
            "'SharedMemory' backend is supported only at Python 3.8+. " "Fail back to 'sharedctypes' backend"
        )
    if _has_SharedMemory and backend == "SharedMemory":
        return SharedMemoryArray(ctype, len_)
    if backend == "sharedctypes":
        return ctypesArray(ctx, ctype, len_)

    raise ValueError(f"Unknown backend: {backend}")


def RawValue(ctx, ctype, init, backend):  # noqa: N802
    """
    Get RawValue for backend

    Parameters
    ----------
    ctx
        Multiprocessing Context
    ctype
        C type
    init
        Init value
    backend : {"SharedMemory", "sharedctypes"}
        Backend

    Raises
    ------
    ValueError
        If ``backend`` is unknown.
    """
    if isinstance(ctx, SyncManager):
        ctx = ctx._ctx  # noqa: SLF001
    if not _has_SharedMemory and backend == "SharedMemory":
        backend = "sharedctypes"
        logger.warning(
            "'SharedMemory' backend is supported only at Python 3.8+. " "Fail back to 'sharedctypes' backend"
        )
    if _has_SharedMemory and backend == "SharedMemory":
        return SharedMemoryValue(ctype, init)
    if backend == "sharedctypes":
        return ctx.Value(ctype, init, lock=False)

    raise ValueError(f"Unknown backend: {backend}")


def try_start(ctx):
    """
    Try start SyncManager Server

    Parameters
    ----------
    ctx
        Multiprocessing Context

    Raises
    ------
    ProcessError
        If ``ctx`` is instance of ``SyncManager`` and it has been shutdowned.

    Notes
    -----
    This function is compatible layer for multiple python versions and contexts.
    """
    if isinstance(ctx, SyncManager):
        if ctx._state.value == State.SHUTDOWN:  # noqa: SLF001
            # Default behavior:
            # - Python 3.6 : Assertion Failuer
            # - Python 3.7+: Raise ProcessError
            raise ProcessError("Manager has shut down")
        if ctx._state.value == State.INITIAL:  # noqa: SLF001
            ctx.start()
