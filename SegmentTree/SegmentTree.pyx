# distutils: language = c++

from SegmentTree cimport SegmentTree

cdef class PySegmentTree:
    def __cinit__(self, size=2,f=lambda a,b: a+b):
        print("Segment Tree")

        self.st = SegmentTree(size,f)

    def __setitem__(self,i,v):
        self.st.set(i,v)

    def __getitem__(self,i):
        return self.st.get(i)
