

### The $J(\sigma)$ Approximation

In the provided paper, the average mutual information is approximated using the $J(\sigma)$ function, while its inverse is denoted as $J^{-1}(I)$. These functions characterize the relationship between the mutual information and the standard deviation of the log-likelihood ratio (LLR) values.

The paper defines these approximations as follows:

For computer implementation, the $J(\sigma)$ function is split into two intervals based on a threshold of $\sigma^{*} = 1.6363$:

$$J(\sigma)\approx\begin{cases}a_{J,1}\sigma^{3}+b_{J,1}\sigma^{2}+c_{J,1}\sigma, & 0\le\sigma\le\sigma^{*}\\ 1-e^{a_{J,2}\sigma^{3}+b_{J,2}\sigma^{2}+c_{J,2}\sigma+d_{J,2}}, & \sigma^{*}<\sigma<10\\ 1, & \sigma > 10\end{cases}$$

**Numerical Constants for $J(\sigma)$:**
* $a_{J,1} = -0.0421061$
* $b_{J,1} = 0.209252$
* $c_{J,1} = -0.00640081$
* $a_{J,2} = 0.00181491$
* $b_{J,2} = -0.142675$
* $c_{J,2} = -0.0822054$
* $d_{J,2} = 0.0549608$

---

### The $J^{-1}(I)$ Approximation
The inverse function is split into two intervals at the threshold $I^{*} = 0.3646$:

$$J^{-1}(I)\approx\begin{cases}a_{\sigma,1}I^{2}+b_{\sigma,1}I+c_{\sigma,1}\sqrt{I}, & 0\le I\le I^{*}\\ -a_{\sigma,2}\ln[b_{\sigma,2}(1-I)]-c_{\sigma,2}I, & I^{*}<I<1\end{cases}$$

**Numerical Constants for $J^{-1}(I)$:**
* $a_{\sigma,1} = 1.09542$
* $b_{\sigma,1} = 0.214217$
* $c_{\sigma,1} = 2.33727$
* $a_{\sigma,2} = 0.706692$
* $b_{\sigma,2} = 0.386013$
* $c_{\sigma,2} = -1.75017$



# Naive MI-based Algorithm

- **CN clusters:** $\mathfrak{C} = \{C_1, C_2, \ldots, C_M\}$

- For a cluster $C_a \in \mathfrak{C}$, let $M_a$ be the number of VNs that are attached to any of the CNs in this cluster.

- Let $\{L_a^{1, i}, L_a^{2, i}, \ldots, L_a^{M_a, i}\}$ be the set of CN-to-VN LLRs in the $i$-th iteration.

- Under the consistent Gaussian assumption, each LLR $L_a^{j, i} \sim \mathcal{N}(\sigma_a^2/2, \sigma_a^2)$ for $j = 1, 2, \ldots, M_a$.

- Estimate variance $\hat{\sigma}_a^2$ using the realizations of $\{L_a^{1, i}, L_a^{2, i}, \ldots, L_a^{M_a, i}\}$, then project on MI denoted by $I_{C_a}^{\ell} = J(\hat{\sigma}_a)$, where $I_{C_a}^{\ell}$ is the MI associated with cluster $C_a$ in the $\ell$-th iteration.

- Recall stored $I_{C_a}^{\ell-1}$ from the $(\ell-1)$-th iteration.

- In the $\ell$-th iteration, schedule cluster $C_a$ with the maximum value of $|I_{C_a}^{\ell} - I_{C_a}^{\ell-1}|$.

# RL based MI algorithm:

State space: Mutual information

Reward: difference in mutual information from previous iteration

Use DQN