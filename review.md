# Decoder Review

## Work done so far

- Implemented cluster decoder
- Tested Random decoding
- Tested RBP decoding


## Algo for cluster Decoder

![Cluster Decoder Algorithm](decode_algp.png)

I have taken this from the paper cited by matlab for layer decoder.
> Hocevar, D.E. "A reduced complexity decoder architecture via layered decoding of LDPC codes." In IEEE Workshop on Signal Processing Systems, 2004. SIPS 2004, 107-112.

I followed this algo as faithfully as i can, however I didnt quite understand eqn 13, so i replaced it with 

> L(q_j) += newR - oldR

Overall, the performance is promising even if not exactly matching with matlab.

|snr | Custom_rr (10000 frames) | Matlab (10000 frames) |
|:--- | :--- | :--- |
|-3| 0.1581| 0.1584|
|-2| 0.1294| 0.1299|
|-1| 0.1014| 0.1015|
|0| 0.0702| 0.0720|
|1| 0.0242| 0.0405|
|2| 0.0006| 0.0405|
|3| 8.23e-07| 0.0405|


## Algo for RBP

I for now implemented node-wise rbp as given in 'Informed Dynamic Scheduling for Belief-Propagation Decoding of LDPC Codes'.

I did this instead of cluster wise rbp because:

- I still have to expose the messages computed by backend to frontend.
- I want to check if the algorithm is replicable as said in the paper first before modifying it.


![node-wise-rbp](rbp_algo.png)


## Algo for mutual information based scheduling

Based on M2I2 scheduling

> Scheduling happens independent to the BP decoder. The scheduler is SNR dependent and LLR independent.


Original algo:

![m2i2-algo](m2i2_algo.png)


I modified it slightly to accomadate clusters:

![mi_algo](mi_algo_1.png)
![mi_algo](mi_algo_2.png)


