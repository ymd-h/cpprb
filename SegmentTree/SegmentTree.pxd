from libcpp.functional cimport function

cdef extern from "SegmentTree.hh" namespace "ymd":
  cdef cppclass SegmentTree:
    SegmentTree(size_t,function[double])
    double get(size_t)
    void set(size_t, double)
    double reduce(size_t, size_t)
