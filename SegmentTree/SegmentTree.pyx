# distutils: language = c++

from SegmentTree cimport SegmentTree

cdef class PySegmentTree:
    cdef SegmentTree *thisptr
    def __cinit__(self, size=2,f=lambda a,b: a+b):
        print("Segment Tree")

        self.thisptr = new SegmentTree(size,f)

    def __setitem__(self,i,v):
        self.thisptr.set(i,v)

    def __getitem__(self,i):
        return self.thisptr.get(i)
