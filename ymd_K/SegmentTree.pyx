# distutils: language = c++

from ymd_K.SegmentTree cimport SegmentTree

cdef class PySegmentTree:
    cdef SegmentTree.SegmentTree[double] *thisptr
    cdef SegmentTree.function[double,double,double] f
    def __cinit__(self, size=2,f=lambda a,b: a+b):
        print("Segment Tree")

        self.f = SegmentTree.function[double,double,double]()
        self.f = [double](double,double)(f)
        self.thisptr = new SegmentTree.SegmentTree[double](size, self.f)

    def __setitem__(self,i,v):
        self.thisptr.set(i,v)

    def __getitem__(self,i):
        return self.thisptr.get(i)
