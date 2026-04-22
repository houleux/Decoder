import numpy as np

class QCLDPCEncoder:
    def __init__(self, base_matrix=None, Z=None, H=None):
        """
        Initializes the Encoder by either lifting the base matrix or using a
        directly provided parity-check matrix, then pre-computing the generator
        matrix for the parity bits.
        
        Args:
            base_matrix (list or np.array): The QC Base Matrix. 
                                            Use -1 for Zero blocks.
                                            Use 0, 1, etc., for shift values.
            Z (int): The lifting factor (size of sub-matrices).
            H (list or np.array, optional): Full parity-check matrix. When
                                           provided, base_matrix and Z are not
                                           required.
        """
        if H is not None:
            self.Z = None
            self.base_matrix = None
            self.Hb = None
            self.mb = None
            self.nb = None
            self.H = np.array(H, dtype=int)

            if self.H.ndim != 2:
                raise ValueError("H must be a 2D matrix")

            self.M, self.N = self.H.shape
            self.K = self.N - self.M

            if self.K <= 0:
                raise ValueError(
                    f"Full matrix H must have more columns than rows for systematic encoding, got {self.M}x{self.N}"
                )

            print(f"Initializing Encoder from H: Full Matrix Size {self.M}x{self.N}, Message Bits: {self.K}")
        else:
            if base_matrix is None or Z is None:
                raise ValueError("Either H or both base_matrix and Z must be provided")

            self.Z = Z
            self.base_matrix = np.array(base_matrix)
            self.Hb = self.base_matrix
            self.mb, self.nb = self.Hb.shape
            
            # Dimensions of the full matrix
            self.M = self.mb * self.Z
            self.N = self.nb * self.Z
            self.K = self.N - self.M
            
            print(f"Initializing Encoder: Full Matrix Size {self.M}x{self.N}, Message Bits: {self.K}")

            # 1. Lift the Base Matrix to get H
            self.H = self._lift_matrix()

        # 2. Split H into Information (Hu) and Parity (Hp) parts
        # We assume the code is systematic: [Message | Parity]
        # H = [Hu | Hp]
        self.Hu = self.H[:, :self.K]
        self.Hp = self.H[:, self.K:]

        # 3. Pre-compute the encoding matrix: P_gen = Hp^(-1) @ Hu
        # This allows us to solve p = P_gen @ u
        print("  > Inverting Parity Matrix (this may take a moment for large Z)...")
        try:
            self.Hp_inv = self._gf2_inverse(self.Hp)
        except ValueError as e:
            raise ValueError("The parity part of the matrix (last M columns) is singular and cannot be inverted. "
                             "This base matrix does not support standard systematic encoding [u | p].") from e
        
        print("  > Computing Generator Matrix...")
        self.P_gen = self._gf2_matmul(self.Hp_inv, self.Hu)
        print("Encoder Ready.")

    def encode(self, message):
        """
        Encodes one or more binary message vectors.
        
        Args:
            message (np.array): 1D array of length K OR 
                               2D array of shape (n_frames, K)
        """
        message = np.array(message, dtype=int)
        
        # Handle 1D input by converting to 2D
        is_1d = (message.ndim == 1)
        if is_1d:
            message = message.reshape(1, -1)

        if message.shape[1] != self.K:
            raise ValueError(f"Message width must be {self.K}, got {message.shape[1]}")

        # Batch Parity Calculation: p = u @ P_gen.T (Modulo 2)
        # We transpose P_gen so we can do (n_frames, K) @ (K, M)
        parity = (message @ self.P_gen.T) % 2
        
        # Concatenate [Message | Parity] along the horizontal axis
        codeword = np.hstack((message, parity))
        
        return codeword.flatten() if is_1d else codeword

    def _lift_matrix(self):
        """Expands the Base Matrix into the full binary Parity Check Matrix."""
        H = np.zeros((self.M, self.N), dtype=int)
        
        for r in range(self.mb):
            for c in range(self.nb):
                shift = self.Hb[r, c]
                if shift != -1:  # -1 indicates a zero block
                    # Create Identity matrix
                    block = np.eye(self.Z, dtype=int)
                    # Cyclically shift columns to the right (standard QC definition)
                    # Note: different standards define shift direction differently.
                    # We use standard right cyclic shift.
                    block = np.roll(block, shift, axis=1)
                    
                    # Place block into H
                    H[r*self.Z : (r+1)*self.Z, c*self.Z : (c+1)*self.Z] = block
        return H

    def _gf2_inverse(self, matrix):
        """Computes the inverse of a matrix over GF(2) using Gaussian Elimination."""
        n = matrix.shape[0]
        # Augment with Identity matrix: [A | I]
        aug = np.hstack((matrix, np.eye(n, dtype=int)))
        
        # Gaussian Elimination
        for i in range(n):
            # Find pivot
            pivot = -1
            for j in range(i, n):
                if aug[j, i] == 1:
                    pivot = j
                    break
            
            if pivot == -1:
                raise ValueError("Matrix is singular")
            
            # Swap rows if necessary
            if pivot != i:
                aug[[i, pivot]] = aug[[pivot, i]]
            
            # Eliminate other rows
            # Vectorized XOR operation for speed
            rows_to_xor = aug[:, i] == 1
            rows_to_xor[i] = False # Don't XOR the pivot row with itself
            aug[rows_to_xor] ^= aug[i]

        # Extact the right half (the inverse)
        return aug[:, n:]

    def _gf2_matmul(self, A, B):
        """Performs Matrix Multiplication over GF(2)."""
        # Standard matmul then modulo 2
        return (A @ B) % 2
