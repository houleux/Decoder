#cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True, embedsignature=True
# distutils: language = c++
import numpy as np
import scipy.sparse
from typing import Optional, List, Union
import warnings
import ldpc.helpers.scipy_helpers

cdef BpSparse* Py2BpSparse(pcm):
    
    cdef int m
    cdef int n
    cdef int nonzero_count

    #check the parity check matrix is the right type
    if isinstance(pcm, np.ndarray) or isinstance(pcm, scipy.sparse.spmatrix):
        pass
    else:
        raise TypeError(f"The input matrix is of an invalid type. Please input\
        a np.ndarray or scipy.sparse.spmatrix object, not {type(pcm)}")

    # Convert to binary sparse matrix and validate input
    pcm = ldpc.helpers.scipy_helpers.convert_to_binary_sparse(pcm)

    # get the parity check dimensions
    m, n = pcm.shape[0], pcm.shape[1]


    # get the number of nonzero entries in the parity check matrix
    if isinstance(pcm,np.ndarray):
        nonzero_count  = int(np.sum( np.count_nonzero(pcm,axis=1) ))
    elif isinstance(pcm,scipy.sparse.spmatrix):
        nonzero_count = int(pcm.nnz)

    # Matrix memory allocation
    cdef BpSparse* cpcm = new BpSparse(m,n,nonzero_count) #creates the C++ sparse matrix object

    #fill sparse matrix
    if isinstance(pcm,np.ndarray):
        for i in range(m):
            for j in range(n):
                if pcm[i,j]==1:
                    cpcm.insert_entry(i,j)
    elif isinstance(pcm,scipy.sparse.spmatrix):
        rows, cols = pcm.nonzero()
        for i in range(len(rows)):
            cpcm.insert_entry(rows[i], cols[i])
    
    return cpcm

cdef coords_to_scipy_sparse(vector[vector[int]]& entries, int m, int n, int entry_count):

    cdef np.ndarray[int, ndim=1] rows = np.zeros(entry_count, dtype=np.int32)
    cdef np.ndarray[int, ndim=1] cols = np.zeros(entry_count, dtype=np.int32)
    cdef np.ndarray[uint8_t, ndim=1] data = np.ones(entry_count, dtype=np.uint8)

    for i in range(entry_count):
        rows[i] = entries[i][0]
        cols[i] = entries[i][1]

    smat = scipy.sparse.csr_matrix((data, (rows, cols)), shape=(m, n), dtype=np.uint8)
    return smat

cdef BpSparse2Py(BpSparse* cpcm):
    cdef int i
    cdef int m = cpcm.m
    cdef int n = cpcm.n
    cdef int entry_count = cpcm.entry_count()
    cdef vector[vector[int]] entries = cpcm.nonzero_coordinates()
    smat = coords_to_scipy_sparse(entries, m, n, entry_count)
    return smat


def io_test(pcm: Union[scipy.sparse.spmatrix,np.ndarray]):
    cdef BpSparse* cpcm = Py2BpSparse(pcm)
    output = BpSparse2Py(cpcm)
    del cpcm
    return output



cdef class BpDecoderBase:

    """
    Bp Decoder base class
    """

    def __cinit__(self,pcm, **kwargs):

        error_rate=kwargs.get("error_rate",None)
        error_channel=kwargs.get("error_channel", None)
        max_iter=kwargs.get("max_iter",0)
        bp_method=kwargs.get("bp_method",0)
        ms_scaling_factor=kwargs.get("ms_scaling_factor",1.0)
        schedule=kwargs.get("schedule", 0)
        omp_thread_count = kwargs.get("omp_thread_count", 1)
        channel_probs = kwargs.get("channel_probs", [None])
        
        # input_vector_type = kwargs.get("input_vector_type", "auto")
        # print(kwargs.get("input_vector_type"))
        # print("input vector type:", input_vector_type)
        
        """
        Docstring test
        """

        cdef int i, j, nonzero_count
        self.MEMORY_ALLOCATED=False

        # Matrix memory allocation
        if isinstance(pcm, np.ndarray) or isinstance(pcm, scipy.sparse.spmatrix):
            pass
        else:
            raise TypeError(f"The input matrix is of an invalid type. Please input\
            a np.ndarray or scipy.sparse.spmatrix object, not {type(pcm)}")
        self.pcm = Py2BpSparse(pcm)
 
        # get the parity check dimensions
        self.m, self.n = pcm.shape[0], pcm.shape[1]

        # allocate vectors for decoder input
        self._error_channel.resize(self.n) #C++ vector for the error channel
        self._syndrome.resize(self.m) #C++ vector for the syndrome

        ## initialise the decoder with default values
        self.bpd = new BpDecoderCpp(self.pcm[0],0,PRODUCT_SUM,PARALLEL,1.0)

        ## set the decoder parameters
        self.bp_method = bp_method
        self.max_iter = max_iter
        self.ms_scaling_factor = ms_scaling_factor
        self.schedule = schedule
        self._omp_thread_count = omp_thread_count
        self._bp_input_type = 'auto'

        ## the ldpc_v1 backwards compatibility
        if isinstance(channel_probs, list) or isinstance(channel_probs, np.ndarray):
            if(len(channel_probs)>0) and (channel_probs[0] is not None):
                error_channel = channel_probs

        # Initialize error channel with default uniform probabilities if not specified
        if error_channel is not None:
            self.error_channel = error_channel
        elif error_rate is not None:
            self.error_rate = error_rate
        else:
            # Default: uniform channel with very low error probability (won't affect LLR-based decoding)
            for i in range(self.n):
                self._error_channel[i] = 0.01

        self.MEMORY_ALLOCATED=True

    def __del__(self):
        if self.MEMORY_ALLOCATED:
            del self.bpd
            del self.pcm

    @property
    def error_rate(self) -> np.ndarray:
        """
        Returns the current error rate vector.

        Returns:
            np.ndarray: A numpy array containing the current error rate vector.
        """
        out = np.zeros(self.n).astype(float)
        for i in range(self.n):
            out[i] = self._error_channel[i]
        return out

    @error_rate.setter
    def error_rate(self, value: Optional[float]) -> None:
        """
        Sets the error rate for the decoder.

        Args:
            value (Optional[float]): The error rate value to be set. Must be a single float value.
        """
        if value is not None:
            if not isinstance(value, float):
                raise ValueError("The `error_rate` parameter must be specified as a single float value.")
            for i in range(self.n):
                self._error_channel[i] = value

    @property
    def error_channel(self) -> np.ndarray:
        """
        Returns the current error channel vector.

        Returns:
            np.ndarray: A numpy array containing the current error channel vector.
        """
        out = np.zeros(self.n).astype(float)
        for i in range(self.n):
            out[i] = self._error_channel[i]
        return out

    @error_channel.setter
    def error_channel(self, value: Union[Optional[List[float]],np.ndarray]) -> None:
        """
        Sets the error channel for the decoder.

        Args:
            value (Optional[List[float]]): The error channel vector to be set. Must have length equal to the block
            length of the code `self.n`.
        """
        if value is not None:
            if len(value) != self.n:
                raise ValueError(f"The error channel vector must have length {self.n}, not {len(value)}.")
            for i in range(self.n):
                self._error_channel[i] = value[i]

    def update_channel_probs(self, value: Union[List[float],np.ndarray]) -> None:
        self.error_channel = value

    @property
    def channel_probs(self) -> np.ndarray:
        out = np.zeros(self.n).astype(float)
        for i in range(self.n):
            out[i] = self._error_channel[i]
        return out


    @property
    def input_vector_type(self)-> str:
        """
        Returns the current input vector type.

        Returns:
            str: The current input vector type.
        """
        return self._bp_input_type


    @input_vector_type.setter
    def input_vector_type(self, input_type: str):
        """
        Sets the input vector type.

        Args:
            input_type (str): The input vector type to be set. Must be either 'syndrome' or 'received_vector'.
        """
        if input_type.lower() in ['auto', 'a', '2']:
            if self.m == self.n:
                raise ValueError("Please specify the input vector type. Either: 1) input_vector_type: 'syndrome' or 2) input_vector_type:\
                'received_vector'.")
            else:
                self._bp_input_type = 'auto'

        elif input_type.lower() in ['syndrome', 's', '0']:
            self._bp_input_type = 'syndrome'
        elif input_type.lower() in ['received_vector', 'r', '1']:
            self._bp_input_type = 'received_vector'
        else:
            raise ValueError(f"The input vector type '{input_type}' is invalid. \
                    Please choose from the following methods: \
                    'input_vector_type=syndrome', 'input_vector_type=received_vector'")


    @property
    def log_prob_ratios(self) -> np.ndarray:
        """
        Returns the current log probability ratio vector.

        Returns:
            np.ndarray: A numpy array containing the current log probability ratio vector.
        """
        out = np.zeros(self.n)
        for i in range(self.n):
            out[i] = self.bpd.log_prob_ratios[i]
        return out

    @property
    def converge(self) -> bool:
        """
        Returns whether the decoder has converged or not.

        Returns:
            bool: True if the decoder has converged, False otherwise.
        """
        return self.bpd.converge

    @property
    def iter(self) -> int:
        """
        Returns the number of iterations performed by the decoder.

        Returns:
            int: The number of iterations performed by the decoder.
        """
        return self.bpd.iterations


    @property
    def check_count(self) -> int:
        """
        Returns the number of rows of the parity check matrix.

        Returns:
            int: The number of rows of the parity check matrix.
        """
        return self.bpd.pcm.m

    @property
    def bit_count(self) -> int:
        """
        Returns the number of columns of the parity check matrix.

        Returns:
            int: The number of columns of the parity check matrix.
        """
        return self.bpd.pcm.n

    @property
    def max_iter(self) -> int:
        """
        Returns the maximum number of iterations allowed by the decoder.

        Returns:
            int: The maximum number of iterations allowed by the decoder.
        """
        return self.bpd.maximum_iterations

    @max_iter.setter
    def max_iter(self, value: int) -> None:
        """
        Sets the maximum number of iterations allowed by the decoder.

        Args:
            value (int): The maximum number of iterations allowed by the decoder.

        Raises:
            ValueError: If value is not a positive integer.
        """
        if not isinstance(value, int):
            raise ValueError("max_iter input parameter is invalid. This must be specified as a positive int.")
        if value < 0:
            raise ValueError(f"max_iter input parameter must be a positive int. Not {value}.")
        self.bpd.maximum_iterations = value if value != 0 else self.n

    @property
    def bp_method(self) -> str:
        """
        Returns the belief propagation method used.

        Returns:
            str: The belief propagation method used. Possible values are 'product_sum' or 'minimum_sum'.
        """
        if self.bpd.bp_method == PRODUCT_SUM:
            return 'product_sum'
        elif self.bpd.bp_method == MINIMUM_SUM:
            return 'minimum_sum'
        else:
            raise ValueError(f"BP method is invalid. \
                    Please choose from the following methods: \
                    'product_sum', 'minimum_sum'")

    @bp_method.setter
    def bp_method(self, value: Union[str,int]) -> None:
        """
        Sets the belief propagation method used.

        Args:
            value (str): The belief propagation method to use. Possible values are 'product_sum' or 'minimum_sum'.

        Raises:
            ValueError: If value is not a valid option.
        """
        if str(value).lower() in ['prod_sum', 'product_sum', 'ps', '0', 'prod sum']:
            self.bpd.bp_method = PRODUCT_SUM
        elif str(value).lower() in ['min_sum', 'minimum_sum', 'ms', '1', 'minimum sum', 'min sum']:
            self.bpd.bp_method = MINIMUM_SUM
        else:
            raise ValueError(f"BP method '{value}' is invalid. \
                    Please choose from the following methods: \
                    'product_sum', 'minimum_sum'")

    @property
    def schedule(self) -> str:
        """
        Returns the scheduling method used.

        Returns:
            str: The scheduling method used. Possible values are 'parallel', 'serial', 'serial_relative', or 'cluster'.
        """
        if self.bpd.schedule == PARALLEL:
            return 'parallel'
        elif self.bpd.schedule == SERIAL:
            return 'serial'
        elif self.bpd.schedule == SERIAL_RELATIVE:
            return 'serial_relative'
        elif self.bpd.schedule == CLUSTER:
            return 'cluster'
        else:
            raise ValueError(f"The BP schedule method is invalid. \
                    Please choose from the following methods: \
                    'schedule=parallel', 'schedule=serial', 'schedule=serial_relative', 'schedule=cluster'")

    @schedule.setter
    def schedule(self, value: Union[str,int]) -> None:
        """
        Sets the scheduling method used.

        Args:
            value (str): The scheduling method to use. Possible values are 'parallel', 'serial', 'serial_relative', or 'cluster'.

        Raises:
            ValueError: If value is not a valid option.
        """
        if str(value).lower() in ['parallel','p','0']:
            self.bpd.schedule = PARALLEL
        elif str(value).lower() in ['serial','s','1']:
            self.bpd.schedule = SERIAL
        elif str(value).lower() in ['serial_relative', 'sr', '2']:
            self.bpd.schedule = SERIAL_RELATIVE
        elif str(value).lower() in ['cluster', 'c', '3']:
            self.bpd.schedule = CLUSTER
        else:
            raise ValueError(f"The BP schedule method '{value}' is invalid. \
                    Please choose from the following methods: \
                    'schedule=parallel', 'schedule=serial', 'schedule=serial_relative', 'schedule=cluster'")

    @property
    def ms_scaling_factor(self) -> float:
        """Get the scaling factor for minimum sum method.

        Returns:
            float: The current scaling factor.
        """
        return self.bpd.ms_scaling_factor

    @ms_scaling_factor.setter
    def ms_scaling_factor(self, value: float) -> None:
        """Set the scaling factor for minimum sum method.

        Args:
            value (float): The new scaling factor.

        Raises:
            TypeError: If the input value is not a float.
        """
        if not isinstance(value, float):
            raise TypeError("The ms_scaling factor must be specified as a float")
        self.bpd.ms_scaling_factor = value

    @property
    def omp_thread_count(self) -> int:
        """Get the number of OpenMP threads.

        Returns:
            int: The number of threads used.
        """
        if self._omp_thread_count != 1:
            warnings.warn("The OpenMP functionality is not yet implemented")
        return self._omp_thread_count

    @omp_thread_count.setter
    def omp_thread_count(self, value: int) -> None:
        """Set the number of OpenMP threads.

        Args:
            value (int): The number of threads to use.

        Raises:
            TypeError: If the input value is not an integer or is less than 1.
        """
        if not isinstance(value, int) or value < 1:
            raise TypeError("The omp_thread_count must be specified as a\
            positive integer.")
        self._omp_thread_count = value
        if self._omp_thread_count != 1:
            warnings.warn("The OpenMP functionality is not yet implemented")

cdef class BpDecoder(BpDecoderBase):
    """
    Belief propagation decoder for binary linear codes.

    This class provides an implementation of belief propagation decoding for binary linear codes. The decoder uses a sparse
    parity check matrix to decode received codewords. The decoding algorithm can be configured using various parameters,
    such as the belief propagation method used, the scheduling method used, and the maximum number of iterations.

    Parameters
    ----------
    pcm : Union[np.ndarray, spmatrix]
        The parity check matrix of the binary linear code, represented as a NumPy array or a SciPy sparse matrix.
    error_rate : Optional[float], optional
        The initial error rate for the decoder, by default None.
    error_channel : Optional[List[float]], optional
        The initial error channel probabilities for the decoder, by default None.
    max_iter : Optional[int], optional
        The maximum number of iterations allowed for decoding, by default 0 (adaptive).
    bp_method : Optional[str], optional
        The belief propagation method to use: 'product_sum' or 'minimum_sum', by default 'minimum_sum'.
    ms_scaling_factor : Optional[float], optional
        The scaling factor for the minimum sum method, by default 1.0.
    schedule : Optional[str], optional
        The scheduling method for belief propagation: 'parallel', 'serial', or 'serial_relative'. By default 'parallel'.
    omp_thread_count : Optional[int], optional
        The number of OpenMP threads to use, by default 1.
    input_vector_type: str, optional
        Use this paramter to specify the input type. Choose either: 1) 'syndrome' or 2) 'received_vector' or 3) 'auto'.
        Note, it is only necessary to specify this value when the parity check matrix is square. When the
        parity matrix is non-square the input vector type is inferred automatically from its length.
    """

    def __cinit__(self, pcm: Union[np.ndarray, scipy.sparse.spmatrix], error_rate: Optional[float] = None,
                 error_channel: Optional[Union[np.ndarray,List[float]]] = None, max_iter: Optional[int] = 0, bp_method: Optional[str] = 'minimum_sum',
                 ms_scaling_factor: Optional[float] = 1.0, schedule: Optional[str] = 'parallel', omp_thread_count: Optional[int] = 1,
                 input_vector_type: str = "auto", **kwargs):

        for key in kwargs.keys():
            if key not in ["channel_probs"]:
                raise ValueError(f"Unknown parameter '{key}' passed to the BpDecoder constructor.")

        self.input_vector_type = input_vector_type
        self._llr_vector.resize(self.n)

        pass

    def __init__(self, pcm: Union[np.ndarray, scipy.sparse.spmatrix], error_rate: Optional[float] = None,
                 error_channel: Optional[Union[np.ndarray,List[float]]] = None, max_iter: Optional[int] = 0, bp_method: Optional[str] = 'minimum_sum',
                 ms_scaling_factor: Optional[float] = 1.0, schedule: Optional[str] = 'parallel', omp_thread_count: Optional[int] = 1,
                 input_vector_type: str = "auto", **kwargs):
        
        pass

    def reset(self):
        """
        Resets the decoder state (iterations, convergence, messages, LLRs).
        """
        self.bpd.reset()

    def initialise_log_domain_bp(self, llr_vector: np.ndarray):
        """
        Initialises the log domain BP with the given LLR vector.
        Sets both initial and current LLRs.
        """
        llr_array = np.ascontiguousarray(llr_vector, dtype=np.float64)
        if llr_array.ndim != 1 or llr_array.shape[0] != self.n:
            raise ValueError(f"The llr_vector must have length {self.n}.")
        
        cdef int i
        for i in range(self.n):
            self._llr_vector[i] = llr_array[i]
            
        self.bpd.initialise_log_domain_bp(self._llr_vector)

    def decode(self, llr_vector: np.ndarray) -> np.ndarray:
        """Decode a vector of per-bit log-likelihood ratios (LLRs).

        Parameters
        ----------
        llr_vector : numpy.ndarray
            A 1D numpy array of length equal to the block length (number of columns).
        """

        llr_array = np.ascontiguousarray(llr_vector, dtype=np.float64)
        if llr_array.ndim != 1 or llr_array.shape[0] != self.n:
            raise ValueError(f"The llr_vector must have length {self.n}. Not length {llr_array.shape[0]}.")

        cdef int i
        cdef bint zero_input_vector = True
        DTYPE = llr_vector.dtype

        for i in range(self.n):
            self._llr_vector[i] = llr_array[i]
            if llr_array[i] != 0.0:
                zero_input_vector = False

        if zero_input_vector:
            self.bpd.converge = True
            return np.zeros(self.bit_count, dtype=DTYPE)

        self.bpd.decode(self._llr_vector)

        out = np.zeros(self.n, dtype=DTYPE)
        for i in range(self.n):
            out[i] = self.bpd.decoding[i]
        return out

    def decode_cluster(self, cluster_checks, llr_vector: Optional[np.ndarray] = None) -> np.ndarray:
        """Run a cluster-local BP update and return the updated LLRs."""

        cluster_array = np.ascontiguousarray(cluster_checks, dtype=np.int32)
        if cluster_array.ndim != 1:
            raise ValueError("cluster_checks must be a 1D array or list of check indices")

        cdef Py_ssize_t cluster_len = cluster_array.shape[0]
        cdef vector[int] c_cluster
        c_cluster.resize(cluster_len)

        cdef Py_ssize_t i
        for i in range(cluster_len):
            idx = int(cluster_array[i])
            if idx < 0 or idx >= self.m:
                raise ValueError(f"cluster_checks[{i}]={idx} is out of range for {self.m} checks")
            c_cluster[i] = idx

        if llr_vector is not None:
            llr_array = np.ascontiguousarray(llr_vector, dtype=np.float64)
            if llr_array.ndim != 1 or llr_array.shape[0] != self.n:
                raise ValueError(f"The llr_vector must have length {self.n}. Not length {llr_array.shape[0]}.")
            
            for i in range(self.n):
                self._llr_vector[i] = llr_array[i]
        else:
            # If no LLR vector is provided, use the current internal LLRs
            # We need to make sure self._llr_vector is up to date with the C++ side
            # The C++ side updates log_prob_ratios in place.
            # We can copy log_prob_ratios to _llr_vector before passing it back.
            for i in range(self.n):
                self._llr_vector[i] = self.bpd.log_prob_ratios[i]

        self.bpd.bp_decode_cluster(self._llr_vector, c_cluster)

        DTYPE = np.float64
        out = np.zeros(self.n, dtype=DTYPE)
        for i in range(self.n):
            out[i] = self.bpd.log_prob_ratios[i]
        return out
        
    @property
    def llr_vector(self) -> np.ndarray:
        """
        Returns the current log probability ratio vector (LLR vector).
        """
        return self.log_prob_ratios

    @llr_vector.setter
    def llr_vector(self, value: np.ndarray):
        """
        Sets the current log probability ratio vector (LLR vector).
        """
        if len(value) != self.n:
            raise ValueError(f"Length mismatch: expected {self.n}, got {len(value)}")
        for i in range(self.n):
            self.bpd.log_prob_ratios[i] = value[i]

    @property
    def decoding(self) -> np.ndarray:
        """
        Returns the current decoded output.

        Returns:
            np.ndarray: A numpy array containing the current decoded output.
        """
        out = np.zeros(self.n).astype(int)
        for i in range(self.n):
            out[i] = self.bpd.decoding[i]
        return out


cdef class SoftInfoBpDecoder(BpDecoderBase):
    """
    A decoder that uses soft information belief propagation algorithm for decoding binary linear codes.

    This class implements a modified version of the belief propagation decoding algorithm that accounts for
    uncertainty in the syndrome readout using a serial belief propagation schedule. The decoder uses a minimum
    sum method as the belief propagation variant. For more information on the algorithm, please see the original
    research paper at https://arxiv.org/abs/2205.02341.

    Parameters
    ----------
    pcm : Union[np.ndarray, spmatrix]
        The parity check matrix for the code.
    error_rate : Optional[float]
        The probability of a bit being flipped in the received codeword.
    error_channel : Optional[List[float]]
        A list of probabilities that specify the probability of each bit being flipped in the received codeword.
        Must be of length equal to the block length of the code.
    max_iter : Optional[int]
        The maximum number of iterations for the decoding algorithm.
    bp_method : Optional[str]
        The variant of belief propagation method to be used. The default value is 'minimum_sum'.
    ms_scaling_factor : Optional[float]
        The scaling factor used in the minimum sum method. The default value is 1.0.
    cutoff : Optional[float]
        The threshold value below which syndrome soft information is used.
    """

    def __cinit__(self, pcm: Union[np.ndarray, spmatrix], error_rate: Optional[float] = None,
                 error_channel: Optional[List[float]] = None, max_iter: Optional[int] = 0, bp_method: Optional[str] = 'minimum_sum',
                 ms_scaling_factor: Optional[float] = 1.0, cutoff: Optional[float] = np.inf, sigma: float = 2.0, **kwargs):

        self.cutoff = cutoff
        if not isinstance(sigma,float) or sigma <= 0:
            raise ValueError("The sigma value must be a float greater than 0.")
        self.sigma = sigma
        self.schedule = "serial"
        self.bp_method = "minimum_sum"
        self.input_vector_type = "syndrome"
        self._soft_syndrome.resize(self.m)

    # def __init__(self, pcm: Union[np.ndarray, spmatrix], error_rate: Optional[float] = None,
    #              error_channel: Optional[List[float]] = None, max_iter: Optional[int] = 0, bp_method: Optional[str] = 'minimum_sum',
    #              ms_scaling_factor: Optional[float] = 1.0, cutoff: Optional[float] = np.inf, sigma: float = 2.0, input_vector_type: str = "syndrome"):

    #     pass

    def decode(self, soft_info_syndrome: np.ndarray) -> np.ndarray:
        """
        Decode the input syndrome using the soft information belief propagation decoding algorithm.

        Parameters
        ----------
        soft_info_syndrome: np.ndarray
            A 1-dimensional numpy array containing the soft information of the syndrome.

        Returns
        -------
        np.ndarray
            A 1-dimensional numpy array containing the decoded output.
        """

        cdef vector[np.float64_t] soft_syndrome
        soft_syndrome.resize(self.m)
        for i in range(self.m):
            soft_syndrome[i] = soft_info_syndrome[i]

        self.bpd.soft_info_decode_serial(soft_syndrome,self.cutoff, self.sigma)

        out = np.zeros(self.n,dtype=np.uint8)
        for i in range(self.n): out[i] = self.bpd.decoding[i]
        return out

    @property
    def soft_syndrome(self) -> np.ndarray:
        """
        Returns the current soft syndrome.

        Returns:
            np.ndarray: A numpy array containing the current soft syndrome.
        """
        out = np.zeros(self.m)
        for i in range(self.m):
            out[i] = self._soft_syndrome[i]
        return out


    @property
    def decoding(self) -> np.ndarray:
        """
        Returns the current decoded output.

        Returns:
            np.ndarray: A numpy array containing the current decoded output.
        """
        out = np.zeros(self.n).astype(int)
        for i in range(self.n):
            out[i] = self.bpd.decoding[i]
        return out






