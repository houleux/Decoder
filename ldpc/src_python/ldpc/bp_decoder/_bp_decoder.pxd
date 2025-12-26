#cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True, embedsignature=True
# distutils: language = c++
from libc.stdlib cimport malloc, calloc, free
from libcpp cimport bool
from libcpp.vector cimport vector
cimport numpy as np
ctypedef np.uint8_t uint8_t

cdef extern from "bp.hpp" namespace "ldpc::bp":
    
    cdef const vector[int] NULL_INT_VECTOR

    cdef enum BpMethod:
        PRODUCT_SUM = 0
        MINIMUM_SUM = 1

    cdef enum BpInputType:
        SYNDROME = 0
        RECEIVED_VECTOR = 1
        AUTO = 2

    cdef enum BpSchedule:
        SERIAL = 0
        PARALLEL = 1
        SERIAL_RELATIVE = 2
        CLUSTER = 3

    cdef cppclass BpEntry "ldpc::bp::BpEntry":
        BpEntry() except +
        bool at_end()

    cdef cppclass BpSparse "ldpc::bp::BpSparse":
        int m
        int n
        BpSparse() except +
        BpSparse(int m, int n, int entry_count) except +
        BpEntry& insert_entry(int i, int j)
        BpEntry& get_entry(int i, int j)
        vector[uint8_t]& mulvec(vector[uint8_t]& input_vector, vector[uint8_t]& output_vector)
        vector[uint8_t] mulvec(vector[uint8_t]& input_vector)

        vector[vector[int]] nonzero_coordinates()
        int entry_count()

        int get_col_degree(int col)
        int get_row_degree(int row)

    cdef cppclass BpDecoderCpp "ldpc::bp::BpDecoder":
            BpDecoderCpp(
                BpSparse& parity_check_matrix,
                int maximum_iterations,
                BpMethod bp_method,
                BpSchedule schedule,
                double min_sum_scaling_factor) except +
            BpSparse& pcm
            int check_count
            int bit_count
            int maximum_iterations
            BpMethod bp_method
            BpSchedule schedule
            double ms_scaling_factor
            vector[uint8_t] decoding
            vector[uint8_t] candidate_syndrome
            vector[double] log_prob_ratios
            vector[double] initial_log_prob_ratios
            int iterations
            bool converge
            vector[uint8_t] decode(vector[double]& llr_vector)
            vector[double] get_residuals()
            vector[double] get_mi_residuals()
            vector[int] m2i2_scheduler(const vector[vector[int]] &P, double code_rate, double EbN0)
            void bp_decode_cluster(const vector[int]& cluster_checks)
            vector[uint8_t] soft_info_decode_serial(vector[double]& soft_syndrome, double cutoff, double sigma)
            # void nodewise_rbp(int max_updates, double residual_eps, bool use_approx_residual)
            void initialise_log_domain_bp(const vector[double] &llr_vector_channel)
            void reset()

cdef class BpDecoderBase:
    cdef BpSparse *pcm
    cdef int m, n
    cdef vector[uint8_t] _syndrome
    cdef vector[double] _error_channel
    cdef bool MEMORY_ALLOCATED
    cdef BpDecoderCpp *bpd
    cdef str user_dtype
    cdef str _bp_input_type
    cdef int _omp_thread_count
    # cdef int random_schedule_seed
    
cdef class BpDecoder(BpDecoderBase):
    cdef vector[double] _llr_vector
    pass

cdef class SoftInfoBpDecoder(BpDecoderBase):
    cdef double sigma
    cdef double cutoff
    cdef vector[double] _soft_syndrome
    pass
