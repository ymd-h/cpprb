# distutils: language = c++

from SegmentTree cimport SegmentTree as cppSegmentTree

cdef class SegmentTree:
    def __cinit__(self, size=2):
        print("Segment Tree")

        self.st = cppSegmentTree(size)

    def __setitem__(i,v):
        st.set(i,v)

    def __getitem__(i):
        return st.get(i)

if __name__ is "main":
    sg = SegmentTree()
