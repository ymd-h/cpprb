# distutils: language = c++

from SegmentTree cimport SegmentTree as cppSegmentTree

cdef class SegmentTree:
    def __cinit__(self, size=2):
        print("Segment Tree")

        self.st = cppSegmentTree(size)

if __name__ is "main":
    sg = SegmentTree()
