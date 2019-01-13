# distutils: language = c++

from ymd_K.SegmentTree cimport SegmentTree

cdef class PySegmentTree:
    def __cinit__(self, size=2,f=lambda a,b: a+b):
        print("Segment Tree")


    def __setitem__(self,i,v):
        self.thisptr.set(i,v)

    def __getitem__(self,i):
        return self.thisptr.get(i)
