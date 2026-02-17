import numpy as np

class QCLDPCEncoder:
    def __init__(self, base_matrix, Z):
        """
        Initializes the Encoder by lifting the base matrix and pre-computing
        the generator matrix for the parity bits.
        
        Args:
            base_matrix (list or np.array): The QC Base Matrix. 
                                            Use -1 for Zero blocks.
                                            Use 0, 1, etc., for shift values.
            Z (int): The lifting factor (size of sub-matrices).
        """
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
        Encodes a binary message vector.
        
        Args:
            message (np.array): 1D binary array of length K (nb-mb)*Z
        
        Returns:
            codeword (np.array): 1D binary array of length N (message + parity)
        """
        message = np.array(message, dtype=int)
        
        if len(message) != self.K:
            raise ValueError(f"Message length must be {self.K}, got {len(message)}")

        # Calculate Parity: p = (Hp^-1 @ Hu) @ u = P_gen @ u
        parity = self._gf2_matmul(self.P_gen, message.reshape(-1, 1)).flatten()
        
        # Concatenate [Message | Parity]
        return np.concatenate((message, parity))

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