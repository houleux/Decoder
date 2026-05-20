# Matrix: H_AB_LDPC_500

## Basic Properties
- **Name**: AB (5,0)-Regular LDPC Code
- **Variable nodes**: 500
- **Check nodes**: 250
- **Code rate**: 1/2
- **Density**: Regular (5,0)

## Type
Regular LDPC code with variable degree 5. Larger version of standard test codes.

## Format
Sparse coordinate (COO) format in CSV

## Characteristics
- Moderate size for experimental studies
- Larger than Mackay (96x48) but smaller than realistic codes
- Good for studying scaling behavior
- Decoding complexity is manageable on typical hardware

## Use Cases
- Medium-scale experiments
- Scalability studies
- Performance evaluation across matrix sizes
- Comparison with larger matrices

## Applications
- Research on iteration depth vs. performance
- Message passing complexity analysis
- Hardware implementation feasibility studies

## File Size
- Matrix file: 13 KB
- Sparsity: ~99.96%

## Comparison
- Larger than: H_AB_3_7_196 (196 vars)
- Smaller than: WRAN (256 vars)
- Comparable to: Practical test codes

## Loading
```python
import pandas as pd
import scipy.sparse as sp

data = pd.read_csv('H_AB_LDPC_500.csv')
h = sp.coo_matrix((
    [1]*len(data),
    (data['row'].values, data['col'].values),
    shape=(250, 500)
)).tocsr()
```
