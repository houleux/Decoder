# Matrix: H_Mackay_96_48

## Basic Properties
- **Name**: Mackay 96x48 Irregular LDPC Code
- **Variable nodes**: 96
- **Check nodes**: 48
- **Code rate**: 1/2
- **Type**: Irregular LDPC code

## Author
David Mackay - designed as a practical irregular LDPC code for testing

## Format
Sparse coordinate (COO) format in CSV

## Characteristics
- Smallest test matrix in our catalog
- Classical reference code widely used in literature
- Well-analyzed performance characteristics
- Fastest decoding times among test matrices
- Ideal for rapid prototyping

## Advantages
- Small enough for interactive debugging
- Large enough to show typical decoder behavior
- Extensive performance data available in literature
- Good for algorithm validation before scaling

## Use Cases
- **Algorithm development**: Quick iteration on new ideas
- **Debugging**: Fast execution for testing
- **Literature comparison**: Extensive reference results available
- **Educational**: Understanding LDPC decoder behavior

## Applications
- Proof-of-concept implementations
- Performance baseline
- Decoder algorithm testing
- RL training (fast episodes)

## Performance Notes
- Decoding latency: Milliseconds (even on CPU)
- Memory requirements: Minimal
- Parallelization benefits: Less pronounced than larger matrices

## References
- MacKay, D. J. (1998). "Good Error-Correcting Codes based on Very Sparse Matrices"
- Extensive performance curves available in original paper
- Used as reference in hundreds of subsequent publications

## File Size
- Matrix file: 1.9 KB
- Sparsity: ~99.97%

## Comparison with Other Matrices
| Matrix | Variables | Checks | File Size | Complexity |
|--------|-----------|--------|-----------|------------|
| Mackay | 96 | 48 | 1.9 KB | 1x (reference) |
| AB 196 | 196 | 98 | 4.3 KB | ~2x |
| AB 500 | 500 | 250 | 13 KB | ~5x |
| WRAN 256 | 256 | ? | 9.9 KB | ~5x |

## Loading
```python
import pandas as pd
import scipy.sparse as sp

data = pd.read_csv('H_Mackay_96_48.csv')
h = sp.coo_matrix((
    [1]*len(data),
    (data['row'].values, data['col'].values),
    shape=(48, 96)
)).tocsr()
```
