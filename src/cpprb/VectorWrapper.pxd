from libcpp.vector cimport vector

cdef class VectorWrapper:
    cdef Py_ssize_t *shape
    cdef Py_ssize_t *strides
    cdef Py_ssize_t itemsize
    cdef size_t ndim
    cdef size_t value_dim
    cdef void update_size(self)
    cdef void set_buffer(self,Py_buffer*)
    cdef void set_shape_strides(self)
    cpdef size_t vec_size(self)

cdef class VectorInt(VectorWrapper):
    cdef vector[int] vec

cdef class VectorDouble(VectorWrapper):
    cdef vector[double] vec

cdef class VectorFloat(VectorWrapper):
    cdef vector[float] vec

cdef class VectorSize_t(VectorWrapper):
    cdef vector[size_t] vec

cdef class PointerDouble(VectorWrapper):
    cdef double* ptr
    cdef size_t _vec_size
    cdef void update_vec_size(self,size_t)
