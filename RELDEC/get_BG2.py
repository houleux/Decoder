import sionna
from sionna.fec.ldpc import LDPC5GEncoder
import scipy.sparse as sp
import numpy as np
import csv

# BG2 is used for short/low-rate codes (k < 292 or rate < 1/4)
# k=100, n=520 → rate ≈ 1/5, so BG2 is selected automatically
enc = LDPC5GEncoder(k=100, n=520)

# pcm is the parity check matrix as a numpy array
H = enc.pcm  # shape (420, 520)

print(f"Shape: {H.shape}")
print(f"NNZ:   {H.sum()}")

# Save as sparse CSV
rows, cols = np.where(H == 1)
with open("H_5GNR_520_100.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["row", "col"])
    for r, c in zip(rows, cols):
        writer.writerow([int(r), int(c)])

print("Saved H_5GNR_520_100.csv")