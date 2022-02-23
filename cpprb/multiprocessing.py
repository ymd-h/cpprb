import atexit
import ctypes
import multiprocessing as mp
import sys

import numpy as np

_has_SharedMemory = sys.version_info >= (3, 8)
if _has_SharedMemory:
    # `SharedMemory` class is prefarable since it can work even after
    # serialization/deserialization by using unique name. This
    # capability allows users to use buffers in Ray (https://ray.io/).
    from multiprocessing.shared_memory import SharedMemory
    class _RawValue:
        def __init__(self, ctx, ctype_, init_=None):
            size = ctypes.sizeof(ctype_)
            self.shm = SharedMemory(create=True, size=size)
            self.dtype = np.dtype(ctype_)

            self.ndarray = np.ndarray((1,), self.dtype, buffer=self.shm.buf)
            if init_ is not None:
                self.ndarray[0] = init_

            self.owner = True

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
            self.owner = False

        def __del__(self):
            self.shm.close()
            if self.owner:
                self.shm.unlink()


def RawArray(ctx, ctype, len, use_SharedMemory=False):
    if _has_SharedMemory and use_SharedMemory:
        size = ctypes.sizeof(ctype) * len
        shm = SharedMemory(create=True, size=size)

        @atexit.register
        def _cleanup():
            shm.close()
            shm.unlink()

        return shm
    else:
        return ctx.Array(ctype, len, lock=False)


def RawValue(ctx, ctype, init, use_SharedMemory=False):
    if _has_SharedMemory and use_SharedMemory:
        return _RawValue(ctx, ctype, init)
    else:
        return ctx.Value(ctype, init, lock=False)
