import ctypes
import multiprocessing as mp
import unittest

import numpy as np

from cpprb.multiprocessing import RawArray, RawValue, ctypesArray, _has_SharedMemory
if _has_SharedMemory:
    from cpprb.multiprocessing import SharedMemoryArray, SharedMemoryValue


class TestCtypesArray(unittest.TestCase):
    def test_array(self):
        cases = [[ctypes.c_float, np.single],
                 [ctypes.c_double, np.double],
                 [ctypes.c_uint8, np.uint8],
                 [ctypes.c_uint16, np.uint16],
                 [ctypes.c_uint32, np.uint32]]

        size = 3

        for c, n in cases:
            with self.subTest(ctype=c):
                array = ctypesArray(mp.get_context(), c, size)
                self.assertEqual(len(array), size)

                array[0:size] = 2
                array[1] = 1
                self.assertEqual(array[1], 1)
                np.testing.assert_equal(array[0:size], np.asarray([2,1,2], dtype=n))


if __name__ == "__main__":
    unittest.main()
