# Matrix: WRAN_irreg_384_256

## Basic Properties
- **Name**: WRAN Irregular LDPC Code
- **Variable nodes**: 256
- **Standards**: IEEE 802.16 (WiMAX)
- **Type**: Irregular LDPC code
- **Typical code rate**: 2/3

## Description
WRAN (Wireless Regional Area Network) LDPC code based on IEEE 802.16e specification. Used in WiMAX and other broadband wireless access systems.

## Standard Reference
- IEEE 802.16-2009: "Air Interface for Fixed Broadband Wireless Access Systems"
- IEEE 802.16e: "Mobile WiMAX" extension

## Matrix Properties
- Irregular degree distribution
- Optimized for wireless channel conditions
- Practical implementation proven in deployments
- Good performance over wide SNR range

## Format
Sparse coordinate (COO) format in CSV

## Use Cases
- **Realistic wireless scenarios**: WRAN represents actual deployed standards
- **Comparison studies**: How does performance compare to test matrices?
- **System simulation**: Realistic codec implementation
- **Practical algorithms**: For real-world system design

## Applications
- WiMAX system simulation
- Broadband wireless access research
- Practical decoder benchmarking
- Standards compliance testing

## Advantages
- Real-world standard (not synthetic test case)
- Proven deployment track record
- Moderate size for efficient simulation
- Well-documented specification

## Comparison
- Smaller than: 5G NR (19968 variables)
- Larger than: Mackay (96 variables)
- Similar size to: AB 500 (500 variables)

## File Size
- Matrix file: 9.9 KB
- Sparsity: Very sparse (typical for LDPC)

## Historical Context
- Widely deployed in WiMAX systems (2000s-2010s)
- Basis for understanding modern LDPC standards
- Bridge between theoretical test cases and 5G codes

## Loading
```python
import pandas as pd
import scipy.sparse as sp

data = pd.read_csv('WRAN_irreg_384_256.csv')
h = sp.coo_matrix((
    [1]*len(data),
    (data['row'].values, data['col'].values)
)).tocsr()
```

## Further Reading
- IEEE 802.16-2009 standard document
- WiMAX deployment case studies
- LDPC optimization for wireless channels
