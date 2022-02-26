import atexit
import ctypes
from logging import getLogger
from multiprocessing.managers import SyncManager
import sys

import numpy as np

logger = getLogger(__name__)

_has_SharedMemory = sys.version_info >= (3, 8)
if _has_SharedMemory:
    # `SharedMemory` class is prefarable since it can work even after
    # serialization/deserialization by using unique name. This
    # capability allows users to use buffers in Ray (https://ray.io/).
    from multiprocessing.shared_memory import SharedMemory, _USE_POSIX
    from multiprocessing.managers import SharedMemoryManager

    def setup_unlink(shm):
        # Work around a resource tracker issues;
        # * Ref: https://bugs.python.org/issue41447
        # * Ref: https://bugs.python.org/issue38119
        if _USE_POSIX:
            from multiprocessing.resource_tracker import unregister
            import _posixshmem

            name = shm._name
            unregister(name, "shared_memory")

            @atexit.register
            def unlink(*args):
                try:
                    _posixshmem.shm_unlink(name)
                except FileNotFoundError:
                    pass


    class SharedMemoryValue:
        def __init__(self, ctype, init=None):
            size = ctypes.sizeof(ctype)
            self.shm = SharedMemory(create=True, size=size)
            setup_unlink(self.shm)

            self.dtype = np.dtype(ctype)
            assert self.dtype.itemsize == size, "BUG: data size mismutch"

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
        def __init__(self, ctype, len):
            self.len = len
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

        def __getslice__(self, start, stop):
            return self.ndarray[start:stop]

        def __setslice__(self, start, stop, values):
            self.ndarray[start:stop] = values


class ctypesArray:
    def __init__(self, ctx, ctype, len):
        if isinstance(ctx, SyncManager):
            ctx = ctx._ctx
        self.shm = ctx.Array(ctype, len, lock=False)
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

    def __getslice__(self, start, stop):
        return self.ndarray[start:stop]

    def __setslice__(self, start, stop, values):
        self.ndarray[start:stop] = values

def RawArray(ctx, ctype, len, backend="sharedctypes"):
    len = int(len)
    if not _has_SharedMemory and backend == "SharedMemory":
        backend = "sharedctypes"
        logger.warning("'SharedMemory' backend is supported only at Python 3.8+. " +
                       "Fail back to 'sharedctypes' backend")
    if _has_SharedMemory and backend == "SharedMemory":
        return SharedMemoryArray(ctype, len)
    elif backend == "sharedctypes":
        return ctypesArray(ctx, ctype, len)
    else:
        raise ValueError(f"Unknown backend: {backend}")


def RawValue(ctx, ctype, init, backend="sharedctypes"):
    if not _has_SharedMemory and backend == "SharedMemory":
        backend = "sharedctypes"
        logger.warning("'SharedMemory' backend is supported only at Python 3.8+. " +
                       "Fail back to 'sharedctypes' backend")
    if _has_SharedMemory and backend == "SharedMemory":
        return SharedMemoryValue(ctype, init)
    elif backend == "sharedctypes":
        if isinstance(ctx, SyncManager):
            ctx = ctx._ctx
        return ctx.Value(ctype, init, lock=False)
    else:
        raise ValueError(f"Unknown backend: {backend}")
