cdef extern from "<functional>" namespace "std" nogil:
    cdef cppclass function[T,T,T]:
        function() except +
        function(T*) except +
        function(function&) except +
        function(void*) except +

        function operator=(T*)
        function operator=(function&)
        function operator=(void*)
        function operator=[U](U,U)

        bint operator bool()

cdef extern from "SegmentTree.hh" namespace "ymd":
  cdef cppclass SegmentTree[T]:
    SegmentTree(size_t,function[T,T,T])
    T get(size_t)
    void set(size_t, T)
    T reduce(size_t, size_t)
