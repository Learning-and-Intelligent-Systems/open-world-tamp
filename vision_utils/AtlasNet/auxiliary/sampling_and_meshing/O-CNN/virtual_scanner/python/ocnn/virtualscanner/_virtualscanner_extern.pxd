# distutils: language = c++

from libcpp cimport bool
from libcpp.string cimport string


cdef extern from "virtual_scanner/virtual_scanner.h" nogil:
    cdef cppclass VirtualScanner:
        bool scanning(const string&, int, bool, bool)
        bool save_binary(const string&)
        bool save_ply(const string&)
