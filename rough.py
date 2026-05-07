# Quick conversion
import csv
import numpy as np

def alist_to_csv(alist_path, csv_path):
    """Convert MacKay ALIST format to sparse CSV."""
    with open(alist_path, "r") as f:
        lines = [l.strip() for l in f if l.strip()]
    
    n, m = map(int, lines[0].split())  # n=48 cols, m=96 rows
    
    # Build sparse matrix from ALIST variable node neighbor lists
    rows, cols = [], []
    for col in range(n):
        neighbors = list(map(int, lines[4 + col].split()))
        for row_1idx in neighbors:
            if row_1idx != 0:  # ALIST uses 1-based indexing
                rows.append(row_1idx - 1)
                cols.append(col)
    
    # Write sparse CSV
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["row", "col"])
        for r, c in zip(rows, cols):
            writer.writerow([r, c])
    
    print(f"Converted {alist_path} → {csv_path}")
    print(f"Matrix: {m}×{n}, non-zeros: {len(rows)}")

alist_to_csv('/root/Research/RithvikDecoder/Decoder/Mackay_96_48.txt',
             '/root/Research/RithvikDecoder/Decoder/RELDEC/matrices/H_Mackay_96_48.csv')