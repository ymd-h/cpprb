from libcpp.functional cimport function

cdef extern from "SegmentTree.hh" namespace "ymd":
  cdef cppclass SegmentTree[T]:
    SegmentTree(size_t,function[T])
    T get(size_t)
    void set(size_t, T)
    T reduce(size_t, size_t)
