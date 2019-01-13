# distutils: language = c++

from SegmentTree cimport SegmentTree as cppSegmentTree

cdef class SegmentTree:
    def __cinit__(self, size=2,f=lambda a,b: a+b):
        print("Segment Tree")

        self.st = cppSegmentTree(size,f)

    def __setitem__(self,i,v):
        self.st.set(i,v)

    def __getitem__(self,i):
        return self.st.get(i)
