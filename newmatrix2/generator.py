import numpy as np

Z = 384

# load base graph shifts
BG = np.loadtxt("NR_2_1_384.txt", dtype=int)

m, n = BG.shape

rows = []
cols = []

for r in range(m):
    for c in range(n):

        shift = BG[r,c]

        if shift >= 0:

            for i in range(Z):

                row = r*Z + i
                col = c*Z + (i + shift) % Z

                rows.append(row)
                cols.append(col)

with open("H_BG2_Z384.txt","w") as f:
    for r,c in zip(rows,cols):
        f.write(f"{r} {c}\n")