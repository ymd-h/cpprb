import numpy as np
import unittest
from ymd_K import SegmentTree

class TestPySegmentTree(unittest.TestCase):
    """ PySegmentTree.py """
    N_buffer_size = 16

    @classmethod
    def setUpClass(cls):
        cls.sg = PySegmentTree.PySegmentTree(cls.N_buffer_size,lambda a,b: a+b)

        for i in range(cls.N_buffer_size):
            cls.sg[i] = i * 1.0
            print("{}: {}".format(i,cls.sg[i]))

    def test_reduce(self):
        for i in range(self.N_buffer_size):
            for j in range(i,self.N_buffer_size):
                self.assertAlmostEqual(0.5*(j-i)*(i*j),self.sg.reduce(i,j)))

if __name__ == '__main__':
    unittest.main()
