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
    from multiprocessing.managers import SharedMemoryManager

    _manager = None

    def get_manager():
        global _manager

        if _manager is None:
            _manager = SharedMemoryManager()
            _manager.start()

            @atexit.register
            def shutdown(*args):
                _manager.shutdown()

        return _manager


    class _RawValue:
        def __init__(self, ctype, init=None):
            size = ctypes.sizeof(ctype)
            self.shm = get_manager().SharedMemory(size=size)
            self.dtype = np.dtype(ctype)

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


    class _RawArray:
        def __init__(self, ctype, len):
            size = ctypes.sizeof(ctype) * len
            self.shm = get_manager().SharedMemory(size=size)

        def __getstate__(self):
            return self.shm

        def __setstate__(self, shm):
            self.shm = shm

        def __len__(self):
            return len(self.shm)

        def __getitem__(self, i):
            return self.shm[i]

        def __setitem__(self, i, value):
            self.shm[i] = value

        def __getslice__(self, start, stop):
            return self.shm[start:stop]

        def __setslice__(self, start, stop, values):
            self.shm[start:stop] = values

        @property
        def buf(self):
            return self.shm.buf



def RawArray(ctx, ctype, len, backend="sharedctypes"):
    if _has_SharedMemory and backend == "SharedMemory":
        return _RawArray(ctype, len)
    else:
        return ctx.Array(ctype, len, lock=False)


def RawValue(ctx, ctype, init, backend="sharedctypes"):
    if _has_SharedMemory and backend == "SharedMemory":
        return _RawValue(ctype, init)
    else:
        return ctx.Value(ctype, init, lock=False)
