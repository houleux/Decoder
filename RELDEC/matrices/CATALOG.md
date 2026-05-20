# LDPC Matrix Catalog

*Last updated: 2026-05-19*

## Summary

This catalog describes all LDPC parity-check matrices used in RELDEC experiments. All matrices are stored in sparse coordinate format (COO) as CSV files, with one matrix per file.

## Matrix Index

### 1. AB (3,7)-Regular LDPC (196 variable nodes)
- **File**: `H_AB_3_7_196.csv`
- **Type**: Regular (3,7) LDPC code
- **Rows (checks)**: 98
- **Columns (variables)**: 196
- **Density**: 0.032 (sparse)
- **Code rate**: 1/2
- **Description**: Commonly used regular (3,7) LDPC code with 196 variable nodes. Standard test case for LDPC decoding algorithms.
- **Applications**: Benchmarking, baseline comparisons

### 2. AB (5,0)-Regular LDPC (500 variable nodes)
- **File**: `H_AB_LDPC_500.csv`
- **Type**: Regular (5,0) LDPC code
- **Rows (checks)**: 250
- **Columns (variables)**: 500
- **Density**: 0.040
- **Code rate**: 1/2
- **Description**: Larger regular LDPC code with 500 variable nodes. Used for scaled-up experiments.
- **Applications**: Medium-size decoding tests, scalability studies

### 3. Mackay 96x48 LDPC
- **File**: `H_Mackay_96_48.csv`
- **Type**: Irregular LDPC code
- **Rows (checks)**: 48
- **Columns (variables)**: 96
- **Density**: 0.070
- **Code rate**: 1/2
- **Description**: Mackay LDPC code designed by David Mackay. Small, classical test case widely used in literature.
- **References**: MacKay, D. J. (1998). "Good Error-Correcting Codes"
- **Applications**: Rapid prototyping, reference baseline

### 4. WRAN 384x256 Irregular LDPC
- **File**: `WRAN_irreg_384_256.csv`
- **Type**: Irregular LDPC code
- **Rows (checks)**: Unknown (sparse matrix)
- **Columns (variables)**: 256 base + lifting
- **Code rate**: 2/3 (estimated)
- **Description**: WRAN (Wireless Regional Area Network) standard irregular LDPC code. Based on IEEE 802.16e specification.
- **Standards**: IEEE 802.16 (WiMAX)
- **Applications**: Realistic wireless decoding scenarios

### 5. 5G NR Base Graph 2 (BG2, Z=384)
- **File**: `H_BG2_Z384.txt`
- **Type**: QC-LDPC (quasi-cyclic)
- **Rows**: 16128 (42 × 384)
- **Columns**: 19968 (52 × 384)
- **Lifting size**: Z = 384
- **Code rate**: 1/3
- **Description**: 5G NR (New Radio) LDPC code using Base Graph 2 with lifting size 384. Generated from shift coefficient matrix.
- **Standards**: 3GPP 5G NR (TS 38.212)
- **Source**: NR_2_1_384.txt (shift coefficient matrix)
- **Applications**: Modern wireless systems, 5G standardization

### 6. PCM 802.16e Regular (Z=48)
- **File**: `PCM_802_16e_R12_z48.csv`
- **Type**: QC-LDPC
- **Lifting size**: Z = 48
- **Code rate**: 1/2
- **Description**: IEEE 802.16e (WiMAX) quasi-cyclic LDPC matrix with lifting size 48. Used in cellular and broadband wireless access.
- **Standards**: IEEE 802.16e (WiMAX)
- **Applications**: WiMAX system simulation, QC-LDPC decoding

## Format Specification

### Sparse Coordinate (COO) Format
All matrices are stored in CSV format with columns:
```
row,col
0,3
1,2
2,1
...
```

Where:
- `row`: Row index (0-based) of non-zero element
- `col`: Column index (0-based) of non-zero element
- Each line represents a 1 in the H matrix at (row, col)

### Loading Matrices

**Python (scipy):**
```python
import pandas as pd
import scipy.sparse as sp

# Load from COO format
data = pd.read_csv('matrix.csv')
h_coo = sp.coo_matrix((
    [1]*len(data),
    (data['row'].values, data['col'].values)
))
h_csr = h_coo.tocsr()
```

## Statistics

- **Total matrices**: 6
- **Storage format**: CSV (sparse coordinate)
- **Sizes**: 96 to 19968 variable nodes
- **Code rates**: 1/3 to 2/3
- **Standards covered**: WRAN, IEEE 802.16e, 5G NR

## Matrix Properties

| Name | Vars | Checks | Density | Rate | Type | File |
|------|------|--------|---------|------|------|------|
| AB (3,7)-196 | 196 | 98 | 0.032 | 1/2 | Regular | H_AB_3_7_196.csv |
| AB (5,0)-500 | 500 | 250 | 0.040 | 1/2 | Regular | H_AB_LDPC_500.csv |
| Mackay 96x48 | 96 | 48 | 0.070 | 1/2 | Irregular | H_Mackay_96_48.csv |
| WRAN 256 | 256 | ? | ? | 2/3 | Irregular | WRAN_irreg_384_256.csv |
| 5G NR BG2 Z384 | 19968 | 16128 | ~0.003 | 1/3 | QC-LDPC | H_BG2_Z384.txt |
| 802.16e Z48 | ? | ? | ? | 1/2 | QC-LDPC | PCM_802_16e_R12_z48.csv |

## Choosing a Matrix

**For rapid prototyping**: Use Mackay 96x48 (smallest, fastest decoding)

**For benchmarking**: Use AB (3,7)-196 or AB (5,0)-500 (standard test cases)

**For realistic scenarios**: Use 5G NR BG2 Z384 or 802.16e (standard wireless codes)

**For exploratory research**: Use WRAN 256 (moderate size, real standard)

## Future Additions

- [ ] Additional 5G NR base graphs (BG1, alternative Z values)
- [ ] Additional IEEE 802.16e rate options
- [ ] DVB-S2/DVB-C2 LDPC codes
- [ ] Custom designed codes for specific SNR ranges

## References

1. MacKay, D. J. (1998). "Good Error-Correcting Codes based on Very Sparse Matrices"
2. IEEE Standard 802.16-2009: "Air Interface for Fixed Broadband Wireless Access Systems"
3. 3GPP TS 38.212: "5G NR Multiplexing and channel coding"
4. IEEE Standard 802.11n: "High-Speed Wireless LAN"
