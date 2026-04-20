IEEE TRANSACTIONS ON COMMUNICATIONS, VOL. 71, NO. 10, OCTOBER 2023

# RELDEC: Reinforcement Learning-Based Decoding of Moderate Length LDPC Codes

Salman Habib, Allison Beemer, and Jörg Kliewer, Senior Member, IEEE

Abstract—In this work we propose RELDEC, a novel approach for sequential decoding of moderate length low-density parity-check (LDPC) codes. The main idea behind RELDEC is that an optimized decoding policy is subsequently obtained via reinforcement learning based on a Markov decision process (MDP). In contrast to our previous work, where an agent learns to schedule only a single check node (CN) within a group (cluster) of CNs per iteration, in this work we train the agent to schedule all CNs in a cluster, and all clusters in every iteration. That is, in each learning step of RELDEC an agent learns to schedule CN clusters sequentially depending on a reward associated with the outcome of scheduling a particular cluster. We also modify the state space representation of the MDP, enabling RELDEC to be suitable for larger block length LDPC codes than those studied in our previous work. Furthermore, to address decoding under varying channel conditions, we propose agile meta-RELDEC (AM-RELDEC) that employs meta-reinforcement learning. The proposed RELDEC scheme significantly outperforms standard flooding and random sequential decoding for a variety of LDPC codes, including codes designed for 5G new radio.

Index Terms—Artificial intelligence, channel coding, reinforcement learning (RL), wireless communication.

# I. INTRODUCTION

LOW-DENSITY parity-check (LDPC) codes, a class of channel codes based on sparse parity-check matrices, are known for their excellent performance over symmetric binary input channels [2], [3], [4]. Not only can optimized LDPC codes be capacity-achieving, but as a result of their eponymous sparsity, they admit low complexity graph-based message-passing decoding algorithms, such as belief propagation (BP) [5]. Indeed, due to their performance and practical implementation, they have recently been standardized for data communication in the 5G cellular new radio standard [6], [7].

Manuscript received 21 August 2022; revised 29 March 2023; accepted 4 July 2023. Date of publication 18 July 2023; date of current version 18 October 2023. This work was supported in part by U.S. NSF grant ECCS-1711056 and the U.S. Army Research Laboratory under Cooperative Agreement Number W911NF-17-2-0183. An earlier version of this paper was presented in part at the 17th International Symposium on Wireless Communication Systems, Berlin, Germany [DOI: 10.1109/ISWCS49558.2021.9562199]. The associate editor coordinating the review of this article and approving it for publication was M. Ardakani. (Corresponding author: Salman Habib.)

Salman Habib and Jörg Kliewer are with the Helen and John C. Hartmann Department of Electrical and Computer Engineering, New Jersey Institute of Technology, Newark, NJ 07102 USA (e-mail: sh383@njit.edu; jkliewer@njit.edu).

Allison Beemer is with the Department of Mathematics, University of Wisconsin-Eau Claire, Eau Claire, WI 54701 USA (e-mail: beemera@uwec.edu).

Color versions of one or more figures in this article are available at https://doi.org/10.1109/TCOMM.2023.3296621.

Digital Object Identifier 10.1109/TCOMM.2023.3296621

It has been shown that the order in which iterative updates are sent across the code's graph edges can have significant effect on the decoder's performance [8], [9], [10], [11]. In particular, prior work has shown that sequential scheduling can reduce the number of iterations needed for convergence, hence decreasing the overall decoder complexity. Our own previous work in [12] and [13] explored the idea of utilizing reinforcement learning (RL) in order to schedule the order of soft-information updates. The current work presents two main contributions: (1) we increase the feasible block length for RL-based learning of check node (CN) scheduling by introducing a new state space (i.e. the set of all possible states) for our learning algorithm and adjusting how the scheduling of CNs is executed. We refer to this scheme as RELDEC (reinforcement learning-based decoding). (2) We apply the improvements of RELDEC to the scenario in which channel parameters may be shifting, introducing a novel meta-learning algorithm that can adapt to new parameters with minimal additional training. We call this scheme agile meta-RELDEC, or AM-RELDEC.

LDPC codes may be represented by bipartite Tanner graphs, derived by viewing a parity-check matrix of the code as an adjacency matrix of a graph [3]. The vertices in one part of this graph are termed variable nodes (VNs), while the vertices in the other part are check nodes (CNs). Standard iterative decoding of LDPC codes is performed by passing soft probabilistic information back and forth across the edges of the Tanner graph (e.g. using the BP algorithm) via flooding: in each iteration, all CNs and VNs are updated simultaneously. In contrast, sequential BP decoding, also referred to as layered decoding, updates nodes individually, or as clusters, in sequence. Sequential scheduling problems are concerned with the optimal order of CN (or VN) updates, with the goal of improving the convergence speed and/or decoding performance with respect to flooding and other sequential schemes. Previous work that utilized calculations at vertices to find an optimal scheduling order includes: scheduling of CNs based on the residual change between iterations of the vertex's incoming messages, referred to as node-wise scheduling, or NS [14]; scheduling of VNs based on a relative residual, called efficient dynamic schedule for layered BP, or EDS-LBP [15]; and VN scheduling based on a vertex's reliability as measured by incoming log-likelihood ratios (LLRs), called a reliability-based layered BP decoder, or RBL-BP [16].

In our own previous work, we showed that RL can be used to improve the performance and convergence speed of sequential LDPC decoders as compared to flooding and previous node-wise scheduling schemes [12], [13]. Generally

0090-6778 © 2023 IEEE. Personal use is permitted, but republication/redistribution requires IEEE permission.

See https://www.ieee.org/publications/rights/index.html for more information.

Authorized licensed use limited to: INTERNATIONAL INSTITUTE OF INFORMATION TECHNOLOGY. Downloaded on April 10,2026 at 19:55:22 UTC from IEEE Xplore. Restrictions apply.

speaking, RL is a subset of machine learning which focuses on learning from interaction via computational means. The learning framework is comprised of a (fictitious) agent who interacts with an environment modeled by a finite Markov decision process (MDP) *[17]*. The goal of the agent is to observe the current state of the MDP and then take an action to alter the state. After each action, the agent is rewarded based on the “value” of the action taken given the state. Over time, the agent learns the best action policy in order to maximize the total reward earned over time, information which is encoded using an action value function. In our work here and in *[12]* and *[13]*, we utilize Q-learning, which is a Monte Carlo-based RL algorithm for learning action values *[18, 19]*. Applications of RL include computer games, self-driving cars, and robotics (e.g., *[20, 21, 22]*). RL has also been applied in the context of communications (see the survey paper *[23]*). Prior applications specific to decoding include *[24]*, which framed the factor graph selection problem of BP-based decoding of polar codes as a multi-armed bandit problem, and *[25]*, in which RL was used to make (hard) bit-flipping decisions in iterative decoding. The authors of *[25]* also suggested that RL may be applied to other decoders. Machine learning-based decoders beyond RL have also been explored: deep learning based on neural networks (NNs) was leveraged for decoding linear codes by learning the noise on the communication channel in, e.g., *[26, 27]*, and *[28]*, and a deep learning framework based on hyper-networks was used for decoding short block length LDPC codes in *[29]*.

Our work in *[12]* and *[13]* differed from other RL decoders in that we used RL for soft information BP decoding as opposed to hard-decision bit-flipping, and we sought to learn a CN update schedule as opposed to a choice of graph on which to perform decoding. More specifically, we used RL to approximate an action value function, which indicates the optimal next CN to schedule based on the current state of the messages being passed in decoder. In *[12]*, we initiated the implementation of RL to schedule soft CN updates in an LDPC BP decoder: we considered the model-free RL methods of computing the Gittins index *[30]* of each CN as well as utilizing Q-learning. In *[13]*, we optimized the clustering of check nodes in order to improve performance, and investigated a model-based RL-NS approach with the aid of Thompson sampling. To the best of our knowledge, these works marked the first time that RL learning had been used to schedule sequential BP decoding with soft information updates.

The first main contribution of the current work is improved sequential decoding performance of longer LDPC codes using a novel RL-based scheme called RELDEC. Here, we seek to sequentially schedule clusters of CNs: in a Tanner graph containing $m$ CNs with each cluster comprised of $z$ CNs, the scheduling problem is modeled as a finite MDP with $\lceil m/z\rceil$ possible actions (i.e. cluster selections). We note that the cluster size $z$ should be selected appropriately to balance the trade-off between decoding performance and throughput. As we will see later, smaller clusters will lead to lower Q-learning complexity and improved performance. Thus, we choose cluster sizes no larger than $2$ in our experiments. RELDEC learns an action value function that determines how beneficial a particular choice of cluster is for optimizing the overall cluster scheduling policy. The learned policy is then incorporated into our sequential LDPC decoding algorithm for inference. A limitation of our previous work was that we were only able to implement our scheduling strategy up to block length 196. This is due to the fact that for MDPs with a large state space, Q-learning can require tremendous computational effort. A multitude of methods for reducing this type of learning complexity have been proposed: for example, partitioning the state space (e.g., *[31, 32]*), imposing a state hierarchy (e.g., *[33]*), or reducing dimensionality (e.g., *[34, 35]*). One of the salient features of RELDEC is a significant reduction in the cardinality of the state space due to a new method of identifying the states of our clusters as compared with our previous RL-NS scheme; this enables the decoding of LDPC codes with block lengths of up to 500 bits. Another distinct feature of our current work is that every cluster is scheduled in each executed iteration. In our previous work, a CN cluster was scheduled in each iteration, without any limitation on the possibility of one cluster being scheduled multiple times before another cluster is scheduled for the first time. This change increases the exploration of the RL agent, resulting in a lower BER in the error floor region. Simulations reveal that RELDEC results in significant decoding gain for fixed bit error rate (BER) compared to traditional BP decoding schemes such as flooding, NS, EDS-LBP, RBL-BP, and a variable-node-based layered decoding scheme proposed in the 5G standardization process in *[36]*. Moreover, RL-based decoders require a smaller number of CN to VN message updates, on average, to achieve these gains.

Leveraging our updated state space and scheduling policy, we turn to our second main contribution, where we investigate the scenario in which the parameters of a channel may shift over space and/or time. Prior scheduling policies (including our own) have focused on determining an optimal fixed scheduling order for one particular channel. However, a fixed update order, even one that is optimized for an initially-observed channel, may not provide the optimal CN scheduling policy when a new signal-to-noise ratio (SNR) is encountered by the decoder. To address such a setting, we propose a novel Q-learning-based meta-learning scheme called agile meta-RELDEC, or AM-RELDEC.

Meta-learning has gained considerable attention in recent years. At a high level, meta-learning algorithms first accumulate experience in solving a variety of tasks, and then adapt for solving related but unseen tasks. During adaptation, the algorithm is expected to perform well on the new tasks given minimal training steps and data. A model-agnostic meta-learning (MAML) algorithm was proposed in *[37]* for a wide range of problems including classification, regression, and RL. The MAML scheme relies on gradient descent for model parameter optimization. In *[38]*, the authors proposed a meta-Q-learning (MQL) scheme, a gradient-based approach suitable for Q-learning-based meta-learning. Meta-learning has also been used to solve problems related to communication systems: for instance, MAML was proposed for designing an optimized demodulator which quickly adapts to various channel conditions after being trained using a small number

HABIB et al.: RELDEC OF MODERATE LENGTH LDPC CODES

of pilot symbols in [39] and [40]. This demodulation scheme is suitable for an internet-of-things (IoT) setting, where the channel conditions may vary between devices. MAML was also implemented in the context of channel coding in [41]: the work's supervised meta-learning scheme learns to map a received noisy signal to an estimation of the transmitted message. While AM-RELDEC, which is described in more detail below, also addresses a decoding problem, we note that the block length of 20 considered in [41] is substantially smaller than the block lengths considered in this work.

In our AM-RELDEC scheme, we learn a global action value function corresponding to CN scheduling using data from a mixture of SNRs, then update this action value function locally given an observed (instantaneous) SNR. Once the global action value function has been optimized, adaptation to a local SNR is achieved in an online fashion, based on a minimal number of additional training vectors. In contrast to RELDEC, the agility of AM-RELDEC allows for dynamic scheduling which can adapt to unseen SNRs with substantially reduced additional training using, e.g., pilot signals. Due to its flexibility, AM-RELDEC is well-suited for a multitude of IoT applications. As an example, consider the setting of smart transportation [42]. In this scenario, the power of a signal received by a vehicle from a road side unit (RSU) changes as the vehicle moves, causing fluctuations in channel SNR. During communication with an RSU, the vehicle's sensor can detect the instantaneous SNR of the channel and, using AM-RELDEC, would be able to quickly adapt the decoder's CN scheduling policy. In addition to this nimbleness, AM-RELDEC inherits the gains displayed by RELDEC.

The paper is organized as follows: background on LDPC codes and reinforcement learning is given in Section II. In Section III, we describe how our CN scheduling policy is learned via RELDEC and then incorporated into our learning-based sequential decoding algorithm. Section IV discusses how our scheduling policy is learned using AM-RELDEC. In Section V, we explain our experimental setup and analyze numerical results, comparing the proposed learning-based decoding schemes to LDPC decoders found in the literature. Section VI concludes the paper.

# II. PRELIMINARIES

# A. Low-Density Parity-Check Codes

An  $[n,k]$  binary linear code is a  $k$ -dimensional subspace of  $\mathbb{F}_2^n$ , and may be defined as the kernel of a binary parity-check matrix  $\mathbf{H} \in \mathbb{F}_2^{m \times n}$ , where  $m \geq n - k$ . The code's block length is  $n$ , and rate is  $(n - \mathrm{rank}(\mathbf{H})) / n$ . The Tanner graph of a linear code with parity-check matrix  $\mathbf{H}$  is the bipartite graph  $G_{\mathbf{H}} = (V \cup C, E)$ , where  $V = \{v_0, \ldots, v_{n-1}\}$  is a set of VNs corresponding to the columns of  $\mathbf{H}$ ,  $C = \{c_0, \ldots, c_{m-1}\}$  is a set of CNs corresponding to the rows of  $\mathbf{H}$ , and edges in  $E$  correspond to 1's in  $\mathbf{H}$  [3]. LDPC codes are a class of highly competitive linear codes defined via sparse parity-check matrices or, equivalently, sparse Tanner graphs [2]; they are amenable to low-complexity graph-based message-passing decoding algorithms, making them ideal for practical applications. BP iterative decoding, considered here, is one such algorithm.

In general, a  $(\gamma ,k)$ -regular LDPC quasi-cyclic (QC) code is defined by a parity-check matrix with constant column and row weights equal to  $\gamma$  and  $k$ , respectively [43]. A  $(\gamma ,p)$  array-based (AB) LDPC code is a type of QC code where  $p$  is prime, and its parity-check matrix possesses a special structure [44]. In particular, for AB codes,

$$
\mathbf {H} (\gamma , p) = \left[ \begin{array}{c c c c c} \mathbf {I} &amp; \mathbf {I} &amp; \mathbf {I} &amp; \dots &amp; \mathbf {I} \\ \mathbf {I} &amp; \sigma &amp; \sigma^ {2} &amp; \dots &amp; \sigma^ {p - 1} \\ \vdots &amp; \vdots &amp; \vdots &amp; \dots &amp; \vdots \\ \mathbf {I} &amp; \sigma^ {\gamma - 1} &amp; \sigma^ {2 (\gamma - 1)} &amp; \dots &amp; \sigma^ {(\gamma - 1) (p - 1)} \end{array} \right], \tag {1}
$$

where  $\sigma^z$  denotes the circulant matrix obtained by cyclically left-shifting the entries of the  $p\times p$  identity matrix  $\mathbf{I}$  by  $z$  (mod  $p$ ) positions. Notice that  $\sigma^0 = \mathbf{I}$ . In this work, lifted LDPC codes are obtained by replacing the non-zero (resp., zero) entries of the parity-check matrix with randomly generated permutation (resp., all-zero) matrices.

Parity-check matrices of LDPC codes designed for 5G new radio (5G-NR) possess a QC structure which enables the nodes in its Tanner graph to be updated in parallel [6]. According to the third generation partnership project (3GPP) standard, 5G-NR LDPC codes can be obtained by lifting two types of QC base graphs, known as BG1 and BG2. Depending on the lifting factor, BG1 can be used for constructing LDPC codes with information bits ranging between 500 and 8448 bits, whereas the BG2 base matrix can be used for obtaining shorter codes with information bits ranging between 40 and 2560 bits [6], [45]. The BG1 (resp., BG2) matrix is used for rates between  $1/3$  and  $8/9$  (resp.,  $1/5$  and  $2/3$ ) [6], [45].

# B. Reinforcement Learning

In an RL problem, an agent (learner) interacts with an environment whose state space can be modeled as a finite MDP [17]. The agent takes actions that alter the state of the environment and receives a reward in return for each action, with the goal of maximizing the total reward in a series of actions. The optimized sequence of actions is obtained by employing a policy which utilizes an action value function to determine how beneficial an action is for maximizing the long-term expected reward. In the remainder of the paper, let  $[x] \triangleq \{0, \dots, x - 1\}$ , where  $x$  is a positive integer. Suppose that an environment allows  $m$  possible actions, and let the random variable  $A_{\ell} \in [[m]]$  with realization  $a$  represent the index of an action taken by the agent during learning step  $\ell \in \{0, \dots, \ell_{\max} - 1\}$ . Let  $S_{\ell}$  with realization  $s^{(\ell)} \in \mathbb{Z}$  represent the current state of the environment before taking action  $A_{\ell}$  and let  $S_{\ell + 1}$  with realization  $s^{(\ell)'}$  represent a new state of the MDP after executing  $A_{\ell}$ . Let a state space  $S$  contain all possible state realizations. Also, let  $R_{\ell} = R(S_{\ell}, A_{\ell}, S_{\ell + 1})$  be the reward yielded at step  $\ell$  after taking action  $A_{\ell}$  in state  $S_{\ell}$  which will yield state  $S_{\ell + 1}$ . Optimal policies for MDPs can be estimated via Monte Carlo techniques, such as Q-learning. The estimated action value function  $Q_{\ell}(S_{\ell}, A_{\ell})$  (Q-function) represents the expected long-term reward an agent obtains after taking action  $A_{\ell}$  in state  $S_{\ell}$ . Learning involves iteratively adjusting the action value function for a specific  $(S_{\ell}, A_{\ell})$  pair, based on previously learned action values for the same state

Authorized licensed use limited to: INTERNATIONAL INSTITUTE OF INFORMATION TECHNOLOGY. Downloaded on April 10,2026 at 19:55:22 UTC from IEEE Xplore. Restrictions apply.

IEEE TRANSACTIONS ON COMMUNICATIONS, VOL. 71, NO. 10, OCTOBER 2023

![img-0.jpeg](img-0.jpeg)
Fig. 1. Illustration of RELDEC's learning framework. In each learning step, a fictitious agent schedules a cluster with index $a$ when the environment state, based on hard-decisioned VN values, is $s$. Once an action is taken, the state of the environment changes from $s_{a}^{(\ell)}$ to $s_{a}^{(\ell)'}$ as the VN values are updated after scheduling, and the agent receives reward $R_{a}$ that indicates the accuracy of the hard-decisions taken by the BP algorithm for each blue VN.

and action pair and the reward $R_{\ell}$ earned from taking that action. The optimal policy guides the agent to select an action in a given state that maximizes the Q-function value for that state.

# III. LEARNING THE SCHEDULING POLICY USING RELDEC

The proposed RELDEC scheme consists of a BP decoding algorithm in which the environment is given by the Tanner graph of the LDPC code. Our objective is to learn an optimized sequence of actions, i.e., the scheduling of individual sets $\mathcal{C}_1,\ldots ,\mathcal{C}_{\lceil m / z\rceil}$ of CNs with the set of CNs $\mathcal{C}_i = \{c_{i,1},\dots ,c_{i,z}\}$ of cardinality $z = |\mathcal{C}_i|$. Herein, $c_{i,j} \in [[m]]$, the index of the $j$-th CN in the $i$-th set, is called a cluster in the remainder of the paper. By $\mathcal{N}(\mathcal{C}_i)$ we refer to all the VNs connected to the $i$-th cluster. A single cluster scheduling step is carried out by sending messages from all CNs of a cluster to their neighboring VNs, and subsequently sending messages from these VNs to their CN neighbors. That is, a selected cluster executes one iteration of localized flooding in each decoding instant. Every cluster is scheduled exactly once within a single decoder iteration. Sequential cluster scheduling is carried out until a stopping condition is reached, or an iteration threshold is exceeded. The learning-based decoder relies on a cluster scheduling policy based on a learned action value function.

The MDP related to the RELDEC learning framework for sequential decoding is shown in Fig. 1. The idea is that scheduling a cluster, represented by the blue CNs, updates the state of the environment. In return, the agent receives a reward which is commensurate with the proportion of correct hard decisions. Here, the state is determined by the union of two quantities: the set of check-node indices belonging to a cluster and a binary vector given by the hard-decisioned beliefs of the VNs associated with the cluster.

We define the action space of the MDP as $\mathcal{A} = \left[\left[\lceil m / z\rceil\right]\right]$; this set has a cardinality of $\lceil m / z\rceil$. For example, for $m = 5$ and $z = 2$, $\mathcal{A} = \{0,1,2\}$. Let $\hat{\mathbf{x}}_a^{(\ell)} = [\hat{x}_{0,a}^{(\ell)},\dots ,\hat{x}_{i_s - 1,a}^{(\ell)}]\in \{0,1\}^{\ell_a}$ denote the state of the MDP after scheduling a cluster with index $a\in \mathcal{A}$ during learning step $\ell$, and let $s_a^{(\ell)}\in [[2^{l_a}]]$ refer to the index of a realization of $\hat{\mathbf{x}}_a^{(\ell)}$. Thus, $s_a^{(\ell)}$ also refers to the state of the MDP during learning step $\ell$, and the state space of a cluster is the set of all possible values of $s_a^{(\ell)}$. At a particular decoder

![img-1.jpeg](img-1.jpeg)
Fig. 2. Example of a cluster-induced subgraph, shown with blue squares (cluster consisting of 2 CNs), edges, and circles (VNs). The corresponding state of the cluster is $\hat{\mathbf{x}}_a^{(\ell)}$

iteration, let the output of a cluster be the binary sequence resulting from hard-decisions on the posterior LLRs computed by the (ordered) neighboring VNs. Since the state space of the clusters are pairwise disjoint, the overall state space $S$ of our MDP contains $\sum_{a\in [\lceil \lceil m / z\rceil ]]}2^{l_a}$ realizations of all the cluster outputs $\hat{\mathbf{x}}_0^{(\ell)},\dots ,\hat{\mathbf{x}}_{[m / z] - 1}^{(\ell)}$, where a realization can be thought of as a (cluster, cluster state) pair. Note that the cluster size $z$ should be selected appropriately to balance the trade-off between decoding performance and throughput. For a cluster size of $z = 1$ we schedule only one CN, and, as we show later in Section V, achieve the best performance. In contrast, $z = m$ leads to a flooding schedule which has the highest throughput, but only a modest performance. In this paper, we focus on improving the performance and choose cluster sizes no larger than 2 in our experiments. To facilitate a general treatment, we use an arbitrary $z\in [[m]]$ in the following. An example of a cluster-induced subgraph for the case $z = 2$, and the corresponding state vector $\hat{\mathbf{x}}_a^{(\ell)}$ is shown in Fig. 2.

Let $\mathbf{x} = [x_0, \ldots, x_{n-1}]$ and $\mathbf{y} = [y_0, \ldots, y_{n-1}]$ represent the transmitted and the received words, respectively, where for each $v \in [[n]]$, $x_v \in \{0, 1\}$ and $y_v = (-1)^{x_v} + z$ with $z \sim \mathcal{N}(0, \sigma^2)$. The posterior LLR of $x_v$ is expressed as $L_v = \log \frac{\mathrm{Pr}(x_v = 0 | y_v)}{\mathrm{Pr}(x_v = 1 | y_v)}$. Let $\hat{L}_{\ell}^{(v)} = \sum_{c \in \mathcal{N}(v)} m_{c \to v}^{(\ell)} + L_v$ be the posterior LLR computed by VN $v$ during iteration $\ell$, where $\mathcal{N}(v)$ denotes the set of neighboring CNs of VN $v$, $\hat{L}_0^{(v)} = L_v$, and $m_{c \to v}^{(\ell)}$ is the message received by VN $v$ from neighboring CN $c$ in iteration $\ell$ computed based on standard BP as

$$
m _ {c \rightarrow v} ^ {(\ell)} = 2 \operatorname {a t a n h} \prod_ {v ^ {\prime} \in \mathcal {N} (c) \backslash v} \tanh  \left(\frac {m _ {c ^ {\prime} \rightarrow c} ^ {(\ell - 1)}}{2}\right). \tag {2}
$$

Here, $\mathcal{N}(c)$ denotes the set of neighboring VNs of $c$, and

$$
m _ {v \rightarrow c} ^ {(\ell)} = \sum_ {c ^ {\prime} \in \mathcal {N} (v) \backslash c} m _ {c ^ {\prime} \rightarrow v} ^ {(\ell)} + L _ {v} \tag {3}
$$

is the message propagated from VN $v$ to CN $c$. Moreover, let $\hat{L}_{\ell}^{(j,a)}$ be the posterior LLR computed during learning step $\ell$ by VN $j$ in the subgraph induced by the cluster with index $a \in [[\lceil m / z \rceil]]$. Hence, $\hat{L}_{\ell}^{(j,a)} = \hat{L}_{\ell}^{(v)}$ if VN $v$ in the Tanner graph is also the $j$-th VN in the subgraph induced by the cluster with index $a$.

Note that in our prior work [13], the RL-NS algorithm scheduled a single CN $a$ per decoding iteration based on its reward $\max_{v\in \mathcal{N}(a)}r_{a\to v}$, where $r_{a\to v}$ is the residual of CN

Authorized licensed use limited to: INTERNATIONAL INSTITUTE OF INFORMATION TECHNOLOGY. Downloaded on April 10,2026 at 19:55:22 UTC from IEEE Xplore. Restrictions apply.

HABIB et al.: RELDEC OF MODERATE LENGTH LDPC CODES
5665

$a$ associated with the edge connecting to VN $v$, computed according to $r_{a \to v} \triangleq |m_{a \to v}' - m_{a \to v}|$. Here, $m_{a \to v}$ is the message sent by CN $a$ to its neighboring VN $v$ in the previous iteration, and $m_{a \to v}'$ is the message that CN $a$ would send to VN $v$ in the current iteration, if scheduled. So, the residual represents a quantity which is associated with each edge of the Tanner graph for each BP iteration. Intuitively, the higher the residual of a CN, the further away that portion of the graph is from convergence. Thus, scheduling a CN with the highest residual (reward) leads to faster and more reliable decoding compared to the flooding scheme. Furthermore, in [13], the state space of the MDP is given by the collection of all sequences representing quantized CN values within a cluster.

In contrast, in the proposed RELDEC scheme we consider a new state space representation of the MDP along with a different reward computation, which allows learned decoding of significantly longer block-length LDPC codes. In RELDEC, after scheduling cluster $a$ during learning step $\ell$, the state of the MDP associated with cluster $a$ is given by its output $\hat{\mathbf{x}}_a^{(\ell)}$ that is obtained by taking hard decisions on the vector of posterior LLRs $\hat{\mathbf{L}}_{\ell,a} = [\hat{L}_{\ell}^{(0,a)}\dots \hat{L}_{\ell}^{(l_a - 1,a)}]$, computed according to

$$
\hat {x} _ {j, a} ^ {(\ell)} = \left\{ \begin{array}{l l} 0, &amp; \text {if} \hat {L} _ {\ell} ^ {(j, a)} \geq 0, \\ 1, &amp; \text {otherwise,} \end{array} \right. \tag {4}
$$

where $k_{\mathrm{max}}$ is the maximum CN degree of the cluster, and $l_a \leq k_{\mathrm{max}} z$ is the number of VNs adjacent to cluster $a$. We call $\hat{\mathbf{x}}_a^{(\ell)}$ the state of cluster $a$: it is comprised of the bits reconstructed by the sequential decoder after scheduling cluster $a$ during iteration $\ell$, i.e., the state of the cluster is a sequence of hard-decision VN values associated with the cluster. The collection of signals $\hat{\mathbf{x}}_0^{(\ell)}, \ldots, \hat{\mathbf{x}}_{(va/z)-1}^{(\ell)}$ at the end of decoder iteration $\ell$ forms the entire state of the MDP associated with our RELDEC scheme.

During the learning phase, RELDEC informs the agent of the current state of the decoder and the reward obtained after performing an action (propagating messages from a cluster to its neighboring VNs). Based on these observations, the agent takes future actions, to enhance the total reward earned, which alters the state of the environment as well as the future reward. Given that the transmitted signal $\mathbf{x}$ is known during the training phase, let $\mathbf{x}_a = [x_{0,a},\dots,x_{l_a - 1,a}]$ be a vector containing the $l_a$ bits of $\mathbf{x}$ that are reconstructed in $\hat{\mathbf{x}}_a^{(\ell)}$ by cluster $a$. Note that is the corresponding state vector after scheduling the check nodes in $a$. In each learning step $\ell$, the reward $R_{a}$ obtained by the agent after scheduling cluster $a$ is defined as

$$
R _ {a} = \frac {1}{l _ {a}} \sum_ {j = 0} ^ {l _ {a} - 1} \mathbb {1} \left(x _ {j, a} = \hat {x} _ {j, a}\right), \tag {5}
$$

where $\mathbb{1}(\cdot)$ denotes the indicator function. Thus, the reward earned by the agent after scheduling cluster $a$ is identical to the probability that the transmitted bits $x_{0,a},\ldots ,x_{l_a - 1,a}$ are correctly reconstructed. This new reward metric, resulting from the modification of the state space representation, differs considerably from the maximum residual of the scheduled CN used as reward for RL-NS in [13].

The action values learned by RELDEC are stored in a table with dimension $\max_{a}(2^{l_a})\times \lceil m / z\rceil$. In comparison, for the RL-NS scheme of [13], the learned action values are stored in a table with dimension $M^z\times \lceil m / z\rceil$, where $M = 4$ is the number of quantization levels used for quantizing the CN values, and $M^z$ is the number of all possible sequences of quantized CN values associated with a cluster. Thus, for the MDP discussed in our previous work the state space cardinality, and hence the learning complexity, grows exponentially with $z$. Furthermore, a moderately large $z$ ($\geq 7$) was chosen to ensure a small number of clusters, since there exist dependencies between clusters (i.e., the state of one cluster may depend on the state of another cluster due to the presence of cycles) which Q-learning cannot take into account. However, due to the modification of the MDP in this work, even a choice of $z = 1$ provides considerable reduction of the state space, and hence the size of the action value table is also significantly reduced. This yields a reduced learning complexity with respect to the RL-NS scheme in [13].

In the following, we discuss the learning approach used by RELDEC for obtaining the optimal CN scheduling policy for a given LDPC code. As the new state space representation generates MDPs with moderately large state space size, we utilize standard Q-learning for determining the optimal cluster scheduling order, where the action value, $Q_{\ell +1}(s_a^{(\ell)},a)$, for choosing cluster $a$ in state $s_a$ is given by

$$
\begin{array}{l} Q _ {\ell + 1} \left(s _ {a} ^ {(\ell)}, a\right) = (1 - \alpha) Q _ {\ell} \left(s _ {a} ^ {(\ell)}, a\right) \\ + \alpha \left(R _ {a} + \beta \max  _ {a ^ {\prime} \in [ | | m / z | ] ]} Q _ {\ell} \left(s _ {a} ^ {(\ell) ^ {\prime}}, a ^ {\prime}\right)\right), \tag {6} \\ \end{array}
$$

where $s_a^{(\ell)'}$ represents the new state of the MDP after taking action $a$ in state $s_a^{(\ell)}$, $0 &lt; \alpha &lt; 1$ is the learning rate, $0 &lt; \beta &lt; 1$ is the reward discount rate, $Q_{\ell + 1}(s_a^{(\ell)}, a)$ is a future action value resulting from action $a$ in the current state $s_a^{(\ell)}$ [18], and $\ell$ is the number of learning steps elapsed after observing the initial state $s_a^{(0)}$, corresponding to a received channel output $\mathbf{L} = [L_0, \dots, L_{n-1}]$, in a learning episode. For each $\ell$, cluster $a$ is selected via an $\epsilon$-greedy approach according to

$$
a = \left\{ \begin{array}{l} \text {selected uniformly at random w.p.} \epsilon \text {from} \mathcal {A}, \\ \pi \left(s _ {a} ^ {(\ell)}\right) \text {selected w.p.} 1 - \epsilon , \end{array} \right. \tag {7}
$$

where $\epsilon$ is the probability of exploration, $\mathcal{A} = \{0,\dots ,\lceil m / z\rceil \}$ is a set of all possible actions, and $\pi (s_a^{(\ell)})$ is an agent's policy for taking an action in state $s_a^{(\ell)}$ expressed as

$$
\pi \left(s _ {a} ^ {(\ell)}\right) = \underset {a \in \left[ \left[ \lceil m / z \rceil \right] \right]} {\arg \max } Q _ {\ell} \left(s _ {a} ^ {(\ell)}, a\right). \tag {8}
$$

Note that $\epsilon$ should be large enough to allow adequate exploration, but not so large which inhibits exploitation, i.e., taking actions according to $\pi(s_{a}^{(\ell)})$. Hence, $\epsilon$ should be chosen carefully to balance this trade-off. The action value function is recursively updated $\ell_{\mathrm{max}}$ times according to (6) after observing the initial state. The goal of Q-learning is to find the optimal policy that maximizes the long-term expected reward in

Authorized licensed use limited to: INTERNATIONAL INSTITUTE OF INFORMATION TECHNOLOGY. Downloaded on April 10,2026 at 19:55:22 UTC from IEEE Xplore. Restrictions apply.

IEEE TRANSACTIONS ON COMMUNICATIONS, VOL. 71, NO. 10, OCTOBER 2023

state $s_a^{(\ell)}$, given by

$$
\pi^ {*} \left(s _ {a} ^ {(\ell)}\right) = \underset {a} {\arg \max } Q ^ {*} \left(s _ {a} ^ {(\ell)}, a\right), \tag {9}
$$

where $Q^{*}(s_{a}^{(\ell)},a)$ is the optimal action value for a given $(s_a^{(\ell)},a)$ pair.

For ties (as in the first iteration of Algorithm 1 for $\ell = 0$ and the initial $\mathbf{L}$), we choose an action uniformly at random from all the maximizing actions. During inference, the optimized cluster scheduling policy of standard Q-learning, $\hat{\pi}(s_{a_i}^{(I)})$, for scheduling the $i$-th cluster during decoder iteration $I$ is expressed as

$$
\hat {\pi} \left(s _ {a _ {i}} ^ {(I)}\right) = \underset {a _ {i} \in \left[ \left[ \lceil m / z \rceil \right] \right] \backslash \left\{a _ {0}, \dots , a _ {i - 1} \right\}} {\arg \max } \hat {Q} \left(s _ {a _ {i}} ^ {(I)}, a _ {i}\right), \tag {10}
$$

where $i \in [[\lceil m / z \rceil]]$, and $a_i$ indicates the cluster index to be scheduled at time instant $i$. Further, $\hat{Q}(s_{a_i}^{(I)}, a_i)$ represents the optimized action value after training has been accomplished, which, as $\ell \to \infty$, approaches the optimal action value $Q^*(s_{a_i}^{(I)}, a_i)$ [46], [17, Sec. 6.4]. The RELDEC scheme, which employs standard Q-learning, is shown in Algorithm 1. The input to this algorithm is a parity-check matrix $\mathbf{H}$ and a set $\hat{\mathcal{L}} = \{\mathbf{L}_0, \dots, \mathbf{L}_{|\hat{\mathcal{L}}| - 1}\}$ containing $|\hat{\mathcal{L}}|$ realizations of $\mathbf{L}$ over which Q-learning is performed. Note that each vector in $\hat{\mathcal{L}}$ corresponds to an SNR selected from a set $\mathcal{S} = \{S_1, \dots, S_K\}$, where $S_i \in \mathbb{R}$ is the $i$-th SNR value, and $K$ is the total number of distinct SNRs considered for training. There are $|\hat{\mathcal{L}}| / K$ LLR vectors in $\hat{\mathcal{L}}$ with the same SNR. For a $\mathbf{L} \in |\hat{\mathcal{L}}|$, the action value function in (6) is recursively updated $\ell_{\mathrm{max}}$ times as shown in Step 20. The output of RELDEC is an optimized cluster scheduling policy $\hat{\pi}(s_{a_i}^{(I)})$.

Once learning ends, we utilize Algorithm 2 for inference. The algorithm inputs are the soft channel information vector $\mathbf{L}$, that corresponds to one of the SNRs in $S$, and a parity-check matrix $\mathbf{H}$ of the LDPC code, and $\hat{\mathbf{L}}_I = [\hat{L}_I^{(0)},\dots,\hat{L}_I^{(n - 1)}]$ is initialized using $\mathbf{L}$. The optimized scheduling policy, $\hat{\pi} (s_{a_i}^{(I)})$, is selected in Step 9 of Algorithm 2 according to (10); i.e., an optimized cluster index is selected for the subsequent BP iteration in Steps 10-32. As outlined above, this cluster index depends both on the graph structure and on the received channel values in the previous BP iterations. The output is the reconstructed signal $\hat{\mathbf{x}}$ obtained after executing at most $I_{\mathrm{max}}$ decoding iterations, or until the stopping condition shown in Step 33 is reached. Once decoding ends, we obtain the fully reconstructed signal estimate $\hat{\mathbf{x}} = [\hat{x}_0,\dots,\hat{x}_{n - 1}]$.

This learning-based sequential decoding scheme can be viewed as a sequential generalized LDPC (GLDPC) decoder when $z &gt; 1$, where BP decoding of a cluster-induced subgraph is analogous to decoding a CN subcode of a GLDPC code. When $z = 1$, each cluster represents a single parity-check code, as is the case in a standard LDPC code. Since the full LDPC Tanner graph is connected and contains cycles, there exist dependencies between the messages propagated by the different clusters of the LDPC code. Consequently, the output of a cluster may depend on messages propagated by previously scheduled clusters. Thus, to improve RL performance for $z &gt; 1$, we ensure that the clusters are chosen to be as independent as possible. The choice of clustering is determined prior to

Algorithm 1 RELDEC
Input: set of channel information vectors $\hat{\mathcal{L}}$, parity-check matrix $\mathbf{H}$
Output: optimized cluster scheduling policy $\hat{\pi}(s_{a_i}^{(I)})$
1 Initialization: $Q_0(s_a^{(0)}, a) \gets 0$ for all $s_a^{(0)}$ and $a$
2 for each $\mathbf{L} \in \hat{\mathcal{L}}$ do
3 $\ell \gets 0$
4 $\hat{\mathbf{L}}_\ell \gets \mathbf{L}$
5 determine initial states of all clusters using (4)
// start of an episode
6 while $\ell &lt; \ell_{\max}$ do
7 select cluster $a$ according to (7)
8 for each CN $c$ in $\mathcal{C}_a$, compute and propagate $m_{c \to v}^{(\ell)} \forall v \in \mathcal{N}(c)$
9 for each VN $v$ in $\mathcal{N}(\mathcal{C}_a)$, compute and propagate $m_{v \to c}^{(\ell)} \forall c \in \mathcal{N}(v)$
10 // determine cluster output
11 foreach VN $v$ in the subgraph of cluster $a$ do
12 if $\hat{L}_\ell^{(v)} \geq 0$ then
13 $\hat{x}_{v,a}^{(\ell)} \gets 0$
14 end
15 else
16 $\hat{x}_{v,a}^{(\ell)} \gets 1$
17 end
18
19 determine index $s_a^{(\ell)'}$ of $\hat{\mathbf{x}}_a$
20 update $R_a$ according to (5)
21 compute $Q_{\ell+1}(s_a^{(\ell)}, a)$ according to (6)
22 $s_a^{(\ell+1)} \gets s_a^{(\ell)'}$
23 $\ell \gets \ell + 1$
24 end

learning using the cycle-maximization method discussed in our previous work [12], [13] and omitted here for space considerations. In short, clusters are selected to maximize the number of cycles in the cluster-induced subgraph to minimize inter-cluster dependencies.

The principal activity of RELDEC is the computation of the Q-table. Therefore, we are interested to determine how the number of Q-table updates, according to (6), scale with the size of the training dataset and the number of learning steps for a given training sample. Based on this criterion, Algorithm 1 reveals that the training complexity scales as $\mathcal{O}(|\hat{\mathcal{L}} |\ell_{\mathrm{max}})$.

# IV. CHANNEL STATE ADAPTABILITY VIA META-LEARNING

## A. Overview

In order to make RELDEC useful in a practical communication scenario, we need to learn the CN scheduling policy in (10) for a mixture of SNR values. However, it is important to note that such an adaptation can only be done for a limited set of channel SNR values, as the number of elements in this set affects training complexity and the average decoder

Authorized licensed use limited to: INTERNATIONAL INSTITUTE OF INFORMATION TECHNOLOGY. Downloaded on April 10,2026 at 19:55:22 UTC from IEEE Xplore. Restrictions apply.

HABIB et al.: RELDEC OF MODERATE LENGTH LDPC CODES

Algorithm 2 Learning-Based Sequential BP Decoding Scheme
Input: channel information L, parity-check matrix H
Output: reconstructed signal  $\hat{\mathbf{x}}$
1 Initialization:
2  $I\gets 0$
3  $m_{c\rightarrow v}\gets 0$  // for all CN to VN messages
4  $m_{v\rightarrow c}\gets L_v$  // for all VN to CN messages
5 if decoder iteration  $I &lt;   I_{\mathrm{max}}$  then
6 foreach cluster with index  $a_i$  do
7 Determine state  $s_{a_i}^{(I)}$
8 end
9 schedule cluster  $a_i$  according to policy  $\hat{\pi} (s_{a_i}^{(I)})$
10 for selected cluster with index  $a_i$  do
11 // decode cluster via flooding
12 foreach CN  $c$  in cluster  $a_i$  do
13 foreach VN  $v\in \mathcal{N}(c)$  do
14 compute according to (2) and propagate  $m_{c\rightarrow v}^{(I)}$
15 end
16 end
17 foreach VN  $v$  in the subgraph of cluster  $a_i$  do
18 foreach CN  $c\in \mathcal{N}(v)$  do
19 compute according to (3) and propagate  $m_{v\rightarrow c}^{(I)}$
20 end
21 end
22 if  $\hat{L}_I^{(v)}\gets \sum_{c\in \mathcal{N}(v)}m_{c\rightarrow v}^{(I)} + L_v$  // update posterior LLR
23 end
24 end
25 end
26 end
27 end
28 end
29 end
30 if  $\hat{L}_V^{(v)}\geq 0$  then
31 if  $\hat{L}_V^{(I)}\geq 0$  then
32 if  $\hat{\chi}_{v,a_i}^{(I)}\gets 0$
33 end
34 end
35 end
36 end

performance for each specific SNR. This limitation becomes especially relevant in the case of a wireless communication scenario where channel SNR values are frequently changing due to mobility and interference. To address this challenge, we require a decoder that can quickly adapt to the current channel state, i.e., the instantaneous SNR value. In this section, we demonstrate how this can be achieved through meta-RL.

A general meta-RL strategy can be outlined as follows. Assume that we have  $K$  different tasks to learn. The optimal

![img-2.jpeg](img-2.jpeg)
Fig. 3. Meta-learning framework illustration for sequential LDPC decoding. The optimal global action value function  $Q(s_{a}^{(t)}, a)$  is learned from a dataset of LLR vectors corresponding to a mixture of SNRs during meta-training. In the adaptation phase, this function is used to learn the local action value function  $Q^{(k)}(s_{a}^{(t)}, a)$  for the  $k$ -th local SNR,  $k \in [[K]]$ , using a small set of additional LLR vectors. The resulting optimal local CN scheduling policy  $\pi_{k}^{*}(s_{a}^{(I)})$  is then applied for sequential LDPC decoding.

task-specific policy is learned in two phases: the meta-training phase and the meta-testing (or adaptation) phase. In the meta-training phase, a global long-term expected reward  $Q(s,a)$  is learned, as described in Section III, using a training set derived from a mixture of tasks. The adaptation to the  $k$ -th task occurs in the meta-testing phase, where  $Q(s,a)$  serves as the initialization. From a few additional task-specific training examples, a task-specific policy  $\pi_k(s) = \arg \max_a Q^{(k)}(s,a)$ ,  $k \in [[K]]$  is learned. Here,  $Q^{(k)}(s,a)$  represents the task-specific long-term expected reward learned in the adaptation phase. Typically, this adaptation takes place offline (see [38]).

The general structure of our proposed meta-RELDEC scheme is illustrated in Fig. 3. Initially, meta-training is conducted for a set of  $K$  different channel SNRs. This process involves learning a global long-term expected reward or action value function,  $Q(s_{a}^{(t)},a)$ , using LLR vectors corresponding to a mixture of these SNR values. Next, a local long-term expected reward or action value function,  $Q^{(k)}(s_{a}^{(t)},a)$ , is learned using LLR vectors corresponding to the  $k$ -th SNR value. These local action values contribute to the enhancement of the global action value  $Q(s_{a}^{(t)},a)$  in subsequent rounds of meta-training. After several iterations, the result of these meta-training rounds is an optimized global policy  $Q^{*}(s_{a}^{(t)},a)$ . The adaptation to the  $k$ -th SNR value is then achieved in an online fashion, based on only a few additional training LLR vectors. In a wireless communication context, these LLRs can be easily obtained from received channel pilot symbols, making them readily available. This leads to an SNR-specific scheduling policy. When the channel SNR changes again, the adaptation process restarts from the beginning.

# B. Meta Reinforcement Learning for Optimal CN Scheduling

In the following, we present a meta-RL method that directly estimates the Q-function by minimizing a Q-learning-based

Authorized licensed use limited to: INTERNATIONAL INSTITUTE OF INFORMATION TECHNOLOGY. Downloaded on April 10,2026 at 19:55:22 UTC from IEEE Xplore. Restrictions apply.

IEEE TRANSACTIONS ON COMMUNICATIONS, VOL. 71, NO. 10, OCTOBER 2023

loss function over mini-batches of MDP instances. This approach enables fast and efficient adaptation to varying channel SNRs by terminating the meta-learning algorithm when the loss is below a certain threshold. We further discuss the requirements for successful Q-learning, the need for a large set of channel information vectors, and the optimization of global and local scheduling policies. Recall that the RL environment is modeled as a finite MDP, which can be seen as a sequence of state, action, and reward transitions (see Fig. 1). These transitions can be expressed as a tuple $(s_{a}^{(\ell)}, a, R_{a}, s_{a}^{(\ell)'})$ specific to each learning step $\ell$. Throughout the paper, we refer to this tuple as an MDP instance. Let $\mathcal{D}(\pi(s_{a}^{(\ell)}), \mathbf{L}) \triangleq (s_{a_{0}}^{(0)}, a_{0}, R_{a_{0}}, s_{a_{0}}^{(0)'}), \ldots, (s_{\epsilon_{\ell_{\max}-1}}^{(\ell_{\max}-1)}, a_{\ell_{\max}-1}, R_{a_{\ell_{\max}-1}}, s_{a_{\ell_{\max}-1}}^{(\ell_{\max}-1)'})$ represent a mini-batch of $\ell_{\max}$ MDP instances. These instances are obtained after taking $\ell_{\max}$ actions $a_{0}, \ldots, a_{\ell_{\max}-1}$ according to policy $\pi(s_{a}^{(\ell)})$, $a \in [[\lceil m/z\rceil]]$, where $s_{a_{0}}^{(0)}$ is the initial state of the MDP after observing $\mathbf{L}$ during a learning episode. For instance, if $\ell_{\max} = 2$, $a_{0} = 1$, and $a_{1} = 2$, the second and third clusters are scheduled in the first two learning steps, and $\mathcal{D}(\pi(s_{a}^{(\ell)}), \mathbf{L}) = (s_{1}^{(0)}, 1, R_{1}, s_{1}^{(0)'}), (s_{2}^{(1)}, 2, R_{2}, s_{2}^{(1)'})$. In comparison to MAML, which uses gradient descent to optimize the model parameters [37], our meta-RL scheme directly estimates the Q-function by minimizing a Q-learning-based loss function over a mini-batch $\mathcal{D}(\pi(s_{a}^{(\ell)}), \mathbf{L})$. We will now discuss a method of deriving this function. To do so, note that the Q-learning update procedure shown in (6) can also be expressed as

$$
Q_{\ell+1}(s_{a}^{(\ell)}, a) = Q_{\ell}(s_{a}^{(\ell)}, a) + \alpha(U_{\ell}(s_{a}^{(\ell)}, a) - Q_{\ell}(s_{a}^{(\ell)}, a)), \tag{11}
$$

where $U_{\ell}(s_{a}^{(\ell)}, a) = R_{a} + \beta \max_{a'} Q_{\ell}(s_{a}^{(\ell)'}, a')$, and $U_{\ell}(s_{a}^{(\ell)}, a) - Q_{\ell}(s_{a}^{(\ell)}, a)$ is the temporal difference (TD) error [17]. As learning iteration $\ell$ increases, $Q_{\ell+1}(s_{a}^{(\ell)}, a)$ approaches $Q^{*}(s_{a}^{(\ell)}, a)$, i.e., the meta-RL scheme converges to the true Q-function $Q^{*}(s_{a}^{(\ell)}, a)$ [17, Sec. 6.4], [46]. Consequently, the TD error approaches 0 as $\ell \to \infty$. Assume that meta-RL is carried out on mini-batch $\mathcal{D}(\pi(s_{a}^{(\ell)}), \mathbf{L})$ during a learning episode. At any given learning step $\ell$, the meta-RL algorithm computes a sum of squared TD errors over this mini-batch, given by

$$
\begin{array}{l}
\mathcal{L}(\mathcal{D}(\pi(s_{a}^{(\ell)}), \mathbf{L})) = \sum_{(s_{a}^{(\ell)}, a, R_{a}, s_{a}^{(\ell)'}) \in \mathcal{D}(\pi(s_{a}^{(\ell)}), \mathbf{L})} (U_{\ell}(s_{a}^{(\ell)}, a) \\
- Q_{\ell}(s_{a}^{(\ell)}, a))^{2}, \tag{12}
\end{array}
$$

which is minimized as the agent takes an action via an $\epsilon$-greedy policy at each learning step. Calculating $\mathcal{L}(\mathcal{D}(\pi(s_{a}^{(\ell)}), \mathbf{L}))$ enables us to terminate the meta-learning algorithm once the loss falls below a certain threshold, resulting in a fast and efficient adaptation phase. This distinguishing feature sets our proposed meta-learning approach apart from the baseline RELDEC scheme presented in Section III. It is important to note that a mini-batch $\mathcal{D}(\pi(s_{a}^{(\ell)}), \mathbf{L})$ contains MDP instances resulting from a single channel output $\mathbf{L}$. However, successful

Q-learning requires a substantial amount of training data, and the MDP instances derived from a single training sample does not provide sufficient exploration for the agent. To address this issue, we perform meta-RL on an adequately large set of channel information (LLR) vectors $\mathcal{L} = \{\mathbf{L}_0, \dots, \mathbf{L}_{|\mathcal{L}| - 1}\}$. Hence, in each episode of the proposed meta-RL algorithm, the Q-learning loss is calculated over a batch $\mathcal{B}(\pi(s_a^{(\ell)}), \mathcal{L}) \triangleq \{\mathcal{D}(\pi(s_a^{(\ell)}), \mathbf{L}_0), \dots, \mathcal{D}(\pi(s_a^{(\ell)}), \mathbf{L}_{|\mathcal{L}| - 1})\}$ of MDP instances, which consists of $|\mathcal{L}|$ mini-batches. The corresponding batch loss $\mathcal{L}(\mathcal{B}(\pi(s_a^{(\ell)}), \mathcal{L}))$ is an aggregate of all mini-batch losses $\mathcal{L}(\mathcal{D}(\pi(s_a^{(\ell)}), \mathbf{L}_0)), \dots, \mathcal{L}(\mathcal{D}(\pi(s_a^{(\ell)}), \mathbf{L}_{|\mathcal{L}| - 1}))$, generated by the $|\mathcal{L}|$ learning episodes. The loss must be minimized to ensure Q-learning convergence. In the following, we will discuss how the output of this loss function is minimized in each local and global learning phase of meta-learning.

Let $\mathcal{B}(\pi(s_a^{(\ell)}), \mathcal{L})$ (resp. $\mathcal{B}_k(\pi_k(s_a^{(\ell_k)}), \mathcal{L}_k)$) denote a batch of MDP instances used for learning the global (resp., local) CN scheduling policy. Here, $\mathcal{L}$ contains LLR vectors corresponding to a mixture of $K$ distinct SNR values, i.e., there are $K$ subsets in $\mathcal{L}$, each containing $|\mathcal{L}| / K$ LLR vectors corresponding to a specific SNR. On the other hand, all LLR vectors in $\mathcal{L}_k$ correspond to a single SNR value. The global optimization problem solved by Q-learning can be described by finding the best global action value (or long-term expected reward) function which minimizes the expected loss. By using (12), this can be expressed as

$$
\begin{array}{l}
Q^{*}(s_{a}^{(\ell)}, a) = \underset{Q_{\ell}(s_{a}^{(\ell)}, a) \in \mathbb{R}}{\arg \min} \mathbb{E}_{\mathbf{L} \in \mathcal{L}}[\mathcal{L}(\mathcal{D}(\pi(s_{a}^{(\ell)}), \mathbf{L}))] \\
= \underset{Q_{\ell}(s_{a}^{(\ell)}, a) \in \mathbb{R}}{\arg \min} \mathbb{E}_{\mathbf{L} \in \mathcal{L}} \left[ \sum_{(s_{a}^{(\ell)}, a, R_{a}, s_{a}^{(\ell)'}) \in \mathcal{D}(\pi(s_{a}^{(\ell)}), \mathbf{L})} \right. \\
\left. \left. (U_{\ell}(s_{a}^{(\ell)}, a) - Q_{\ell}(s_{a}^{(\ell)}, a))^{2} \right]. \right]. \tag{13}
\end{array}
$$

The optimal global CN scheduling policy, $\pi^{*}(s_{a_{i}}^{(I)})$, obtained at the end of the global learning phase is defined as

$$
\pi^{*}(s_{a_{i}}^{(I)}) \triangleq \underset{a_{i} \in [[\lceil m/z\rceil]] \setminus \{a_{0}, \dots, a_{i-1}\}}{arg \max} Q^{*}(s_{a_{i}}^{(I)}, a_{i}), \tag{14}
$$

where $I$ is the decoder iteration during inference.

Likewise, finding the best $k$-th local action value or reward function for the $k$-th SNR value can be described by the following optimization problem:

$$
\begin{array}{l}
Q^{(k)*}(s_{a}^{(\ell)}, a) = \underset{Q_{i}^{(k)}(s_{a}^{(\ell)}, a) \in \mathbb{R}}{\arg \min} \mathbb{E}_{\mathbf{L}' \in \mathcal{L}_{k}}[\mathcal{L}(\mathcal{D}_{k}(\pi_{k}(s_{a}^{(\ell)}), \mathbf{l}'))], \\
k \in [[K]]. \tag{15}
\end{array}
$$

Here, $\mathbf{L}' \in \mathcal{L}_k = \{\mathbf{L}_0^{(k)'}, \dots, \mathbf{L}_{|\mathcal{L}_k| - 1}^{(k)'}\}$ is an LLR vector taken from the set $\mathcal{L}_k$ for the $k$-th SNR value and $\mathcal{D}_k(\pi_k(s_a^{(\ell)}), \mathbf{L}')$ is the corresponding mini-batch of MDP instances for $k \in [[K]]$. Similarly to the global policy, the optimal $k$-th local CN scheduling policy $\pi_k^*(s_{a_i}^{(I)})$ used for inference is then given as

$$
\pi_{k}^{*}(s_{a_{i}}^{(I)}) \triangleq \underset{a_{i} \in [[\lceil m/z\rceil]] \setminus \{a_{0}, \dots, a_{i-1}\}}{arg \max} Q^{(k)*}(s_{a_{i}}^{(I)}, a_{i}), \tag{16}
$$

Authorized licensed use limited to: INTERNATIONAL INSTITUTE OF INFORMATION TECHNOLOGY. Downloaded on April 10,2026 at 19:55:22 UTC from IEEE Xplore. Restrictions apply.

HABIB et al.: RELDEC OF MODERATE LENGTH LDPC CODES

where $Q^{(k)*}(s_a^{(I)},a)$ is given by (15) for scheduling the $i$-th CN $a_i$ in each cluster.

In the remainder of the paper, we use simplified notations $\mathcal{B}$ and $\mathcal{B}_k$ to represent the MDP batches $\mathcal{B}(\pi(s_a^{(\ell)}),\mathcal{L})$ and $\mathcal{B}_k(\pi_k(s_a^{(\ell)}),\mathcal{L}_k)$, respectively. Additionally, we use simplified notations $\mathcal{D}_{\mathbf{L}}$ and $\mathcal{D}_{\mathbf{L}'}$ for $\mathcal{D}(\pi(s_a^{(\ell)}),\mathbf{L})$ and $\mathcal{D}_k(\pi_k(s_a^{(\ell)}),\mathbf{L}')$. Since searching over all possible action values for a given $(s_a^{(\ell)},a)$ pair in (13) and (15) is computationally prohibitive, an alternative approach is to employ standard Q-learning. This method yields optimized global and $k$-th local CN scheduling policies by iteratively minimizing empirical losses $\mathcal{L} = \frac{1}{|\mathcal{B}|}\mathcal{L}(\mathcal{B})$ and $\mathcal{L}_k = \frac{1}{|\mathcal{B}_k|}\mathcal{L}(\mathcal{B}_k)$, respectively, during each learning episode.

Subsequently, a global (resp., local) policy update refers to learning the CN scheduling policy by minimizing $\mathcal{L}$ (resp., $\mathcal{L}_k$). Although the distribution of the reward $R_{a}$ for scheduling cluster $a$ may differ across tasks, we consider the tasks related since the environment (the sequential BP decoder) remains unchanged as learning progresses. In the following, we introduce our meta-RL scheme AM-RELDEC, which involves a bi-level policy optimization, as a novel extension of the MAML scheme [37] applied to Q-learning. This approach learns a global CN scheduling policy using a dataset corresponding to an SNR mixture and $K$ local CN scheduling policies based on $K$ separate datasets corresponding to individual SNRs, interactively.

## C. AM-RELDEC Algorithm: Learning the Global Policy

This subsection introduces the AM-RELDEC scheme as shown in Algorithm 3. This novel meta-learning scheme, which has not been published in the open literature to our knowledge, is well suited for wireless communications scenarios with varying channel conditions due to its agility. The global policy for CN scheduling can rapidly adapt online to any local policy corresponding to a particular SNR during the decoding phase. The scheme takes $\mathcal{L}$, $\mathcal{L}_k$, and $\mathbf{H}$ as inputs and outputs an optimized global policy, which serves as a starting point for optimizing the $k$-th local policy during online adaptation, $k \in [[K]]$.

Initially, the $K$ local learning stages of AM-RELDEC are completed in Steps 4-31. In Step 5, the agent initializes a local CN scheduling policy $\pi_k(s_a^{(\ell)})$ using a global CN scheduling policy $\pi(s_a^{(\ell)})$ at the beginning of the $k$-th adaptation phase. Between Steps 6-28, the agent then optimizes this policy by minimizing the sum of squared TD errors for batch $\mathcal{B}_k$ of MDPs, taking actions according to an $\epsilon$-greedy policy in Step 10:

$$
a = \left\{ \begin{array}{l} \text{selected uniformly at random w.p. } \epsilon \text{ from } \left[ \left[ \lceil m / z \rceil \right] \right], \\ f \left(s _ {a} ^ {(\ell)}\right) \text{ selected w.p. } 1 - \epsilon , \end{array} \right. \tag {17}
$$

where $f(s_{a}^{(\ell)}) = \pi_{k}(s_{a}^{(\ell)})$. For global policy optimization, the agent invokes Algorithm 4 in Step 32. In Step 1 of this algorithm, the agent initializes $\pi(s_{a}^{(\ell)})$ using the learned local policies and optimizes it by minimizing the squared TD error for the global batch $\mathcal{B}$ in Steps 2-13. This procedure is repeated

in every meta-learning iteration starting in Step 2 of Algorithm 3. Upon completion of learning, an optimized version of the global policy $\hat{\pi}(s_a^{(I)})$ and an optimized version of the $k$-th local policy $\hat{\pi}_k(s_a^{(I)})$, are obtained.

Learning is carried out through a full BP iteration, as demonstrated in Step 11, based on the chosen cluster $a$ in Step 10 of Algorithm 3. The Q-learning error $\mathcal{L}(\mathcal{D}_{\mathbf{L}^{\prime}})$ is updated in Step 20 after every $x \ll \ell_{\max}$ training steps in each learning episode, where a mini-batch $\mathcal{D}_{\mathbf{L}^{\prime}} \subset \mathcal{B}_k$ contains MDP instances corresponding to a training sample $\mathbf{L}^{\prime}$. This update may result in an overall loss $\mathcal{L}_k$ that is smaller than a threshold $\mathcal{L}_{\min}$. If this occurs, learning can proceed to the next training example before completing all $\ell_{\max}$ learning steps for the current example. The local policy $\pi_k(s_a^{(\ell)})$, optimized in Step 29 as $Q_{\ell}(s_a^{(\ell)}, a)$, is iteratively updated in Step 16 by taking actions in Step 10 according to (17). These Q-learning updates decrease the local loss $\mathcal{L}_k$ in Step 23.

The global policy for a mixture of $K$ SNRs is optimized in Steps 2-13 of Algorithm 4 (called from Step 32 of Algorithm 3). In Step 1, the action value function corresponding to the global policy $\pi(s_{a}^{(\ell)})$ is determined by calculating the average action values across all $K$ local action value functions. In each learning episode, global learning undergoes $\ell_{\mathrm{max}}$ training rounds as shown in Steps 4-9. Consequently, for the training example $\mathbf{L} \in \mathcal{L}$, the cardinality of the corresponding minibatch $\mathcal{D}_{\mathbf{L}} \subset \mathcal{B}$ is $\ell_{\mathrm{max}}$. As the agent continuously interacts with the environment, the global action value $Q_{\ell + 1}(s_{a}^{(\ell)}, a)$ is updated, and $\pi(s_{a}^{(\ell)})$ is optimized by taking actions according to (17), where $f(s_{a}^{(\ell)}) = \pi(s_{a}^{(\ell)})$. This process leads to gradual reduction of the global loss $\mathcal{L}(\mathcal{B})$. Once all meta-iterations are completed, Algorithm 3 generates an optimized global policy $\hat{\pi}(s_{a}^{(I)})$. Note that the complexity of AM-RELDEC, as seen from Algorithms 3 and 4 is $\mathcal{O}((|\mathcal{L}| + K|\mathcal{L}_k|)\ell_{\mathrm{max}})$.

## D. AM-RELDEC Algorithm: Inference and Online Adaptation

Assume that a global policy for a mixture of $K$ SNR values has been already learned through Algorithms 3 and 4 and stored at the decoder. Now, at decoding time, assume that the $k$-th SNR, $k \in [[K]]$, is observed at the channel output. This section addresses how, at decoding time, an optimal CN scheduling is obtained by online adaptation to the observed SNR value. The global policy is used to initialize the online adaptation. We focus on a wireless communication setting where the channel is estimated accurately using pilot signals. The estimation results in a set of LLR vectors $\mathcal{L}_k^*$, with each vector $\mathbf{L}' \in \mathcal{L}_k'$ corresponding to a new SNR value. The SNR values are used to retrain the scheduling policy in the online adaptation shown in Algorithm 5.

This algorithm takes as inputs the LLR values in $\mathcal{L}_k^{\prime}$. The other input comprises the action values $Q_{0}(s_{a}^{(0)},a)$ for all $s_a^{(0)},a$, which are set according to the global policy learned earlier. In Step 1 of Algorithm 5, local action values $Q^{(k)}(s_a^{(0)},a)$ for all $s_a^{(0)},a$ are initialized using the global action values. Steps 2-4 update the local policy with all the training samples in $\mathcal{L}_k^{\prime}$. In particular, Step 3 adapts the $k$-th local policy using Steps 7-27 of Algorithm 3. Note, when

Authorized licensed use limited to: INTERNATIONAL INSTITUTE OF INFORMATION TECHNOLOGY. Downloaded on April 10,2026 at 19:55:22 UTC from IEEE Xplore. Restrictions apply.

IEEE TRANSACTIONS ON COMMUNICATIONS, VOL. 71, NO. 10, OCTOBER 2023

Algorithm 3 AM-RELDEC
Input: set of LLR vectors  $\mathcal{L}$ ,  $\mathcal{L}_k$ , parity-check matrix  $\mathbf{H}$
Output: optimized global scheduling policy  $\hat{\pi}(s_{a_i}^{(I)})$
1  $U_0(s_a^{(0)}, a) \gets 0$ ,  $Q_0(s_a^{(0)}, a) \gets 0$ ,  $\forall s_a^{(0)}, a$ ,  $\mathcal{B} \gets \emptyset$
2 while not done do
// meta-learning phase
3  $k \gets 1$
4 while  $k \leq K$  do
// adapt to the  $k$ -th SNR
5  $\mathcal{B}_k \gets \emptyset$ ,  $Q^{(k)}(s_a^{(0)}, a) \gets Q_0(s_a^{(0)}, a) \forall s_a^{(0)}, a$
// start of an episode
6 for each new  $\mathbf{L}' \in \mathcal{L}_k$  do
7  $\ell \gets 0$ ,  $\hat{\mathbf{L}}_\ell \gets \mathbf{L}'$ ,  $\mathcal{D}_{\mathbf{L}'} \gets \emptyset$ ,  $\mathcal{L}_k \gets 1$
8 determine initial states of all clusters using (4)
9 while  $\mathcal{L}_k &gt; \mathcal{L}_{\min}$  and  $\ell &lt; \ell_{\max}$  do
10 select cluster  $a$  according to (17)
11 decode cluster induced subgraph
12 according to Steps 8-9 of Algorithm 1
13 determine  $\hat{\mathbf{x}}_a$  using Steps 10-17 of Algorithm 1
14 determine index  $s_a^{(\ell)'}$  of  $\hat{\mathbf{x}}_a$  via binary to decimal conversion
15 update  $R_a$  according to (5)
16  $U_\ell(s_a^{(\ell)}, a) \gets R_a + \beta \max_{a' \in [[m/z]]]} Q_\ell(s_a^{(\ell)'}, a')$
17 compute  $Q_{\ell+1}(s_a^{(\ell)}, a)$  according to (6)
18  $s_a^{(\ell+1)} \gets s_a^{(\ell)'}$
19  $\mathcal{D}_{\mathbf{L}'} \gets \mathcal{D}_{\mathbf{L}'} \cup (s_a^{(\ell)}, a, R_a, s_a^{(\ell)'})$
20 for every  $x$  new MDP instances in  $\mathcal{D}_{\mathbf{L}'}$
21 do
22  $\mathcal{L}(\mathcal{D}_{\mathbf{L}'}) \gets \sum_{(s_a^{(\ell)}, a, R_a, s_a^{(\ell)'}) \in \mathcal{D}_{\mathbf{L}'}}(U_\ell(s_a^{(\ell)}, a) - Q_\ell(s_a^{(\ell)}, a))^2$
23  $\mathcal{B}_k \gets \mathcal{B}_k \cup \mathcal{D}_{\mathbf{L}'}$
24  $\mathcal{L}(\mathcal{B}_k) \gets \mathcal{L}(\mathcal{B}_k) + \mathcal{L}(\mathcal{D}_{\mathbf{L}'})$
25  $\mathcal{L}_k \gets \frac{1}{|\mathcal{B}_k|}\mathcal{L}(\mathcal{B}_k)$  // local error minimized as learning continues
26 end
27  $\ell \gets \ell + 1$
28 end
29  $Q_0(s_a^{(\ell)}, a) \gets Q_\ell(s_a^{(\ell)}, a) \forall s_a^{(\ell)}, a$
30 end
31 end
32 perform Steps 1-13 of Algorithm 4 // global policy update
33 end

$\mathcal{L}_k \leq \mathcal{L}_{\min}$  in Step 9 of Algorithm 3, an episode terminates before completing all  $\ell_{\max}$  iterations, accelerating the online adaptation phase.

Algorithm 4 AM-RELDEC (Continued From Step 32 of Algorithm 3)
1 [h]
2  $Q_{0}(s_{a}^{(0)},a)\gets \frac{1}{K}\sum_{k = 1}^{K}Q^{(k)}(s_{a}^{(0)},a)\forall s_{a}^{(0)},a$  // initializes  $\pi (s_a^{(\ell)})$  // start of an episode
3 for each new  $\mathbf{L}\in \mathcal{L}$  do
4  $\ell \leftarrow 0,\hat{\mathbf{L}}_{\ell}\leftarrow \mathbf{L},\mathcal{D}_{\mathbf{L}}\leftarrow \emptyset$
5 while  $\ell &lt;  \ell_{\mathrm{max}}$  do
6 select cluster  $a$  according to 17)
7 repeat Steps 11-17 of Algorithm 3
8  $\mathcal{D}_{\mathbf{L}}\gets \mathcal{D}_{\mathbf{L}}\cup (s_a^{(\ell)},a,R_a,s_a^{(\ell)'})$
9  $\ell \leftarrow \ell +1$
10 end
11  $\mathcal{L}(\mathcal{D}_{\mathbf{L}})\gets$ $\begin{array}{r}\sum_{(s_a^{(\ell_{\mathrm{max}})},a,R_a,s_a^{(\ell_{\mathrm{max}})})}\in \mathcal{D}_{\mathbf{L}}(U_{\ell_{\mathrm{max}}}(s_a^{(\ell_{\mathrm{max}})},a) - Q_{\ell_{\mathrm{max}}}(s_a^{(\ell_{\mathrm{max}})},a))^2 \end{array}$
12  $\mathcal{B}\gets \mathcal{B}\cup \mathcal{D}_{\mathbf{L}}$
13  $Q_{0}(s_{a}^{(\ell_{\mathrm{max}})},a)\gets Q_{\ell_{\mathrm{max}}}(s_{a}^{(\ell_{\mathrm{max}})},a)\forall s_{a}^{(\ell_{\mathrm{max}})},a$
14  $\mathcal{L}\leftarrow \frac{1}{|\mathcal{B}|}\mathcal{L}(\mathcal{B})$  // global error minimized as learning continues
15 end

Algorithm 5 AM-RELDEC (Online Adaptation Phase)
1 [h]
Input: set of LLR vectors  $\mathcal{L}_k^\prime$  obtained after channel estimation, parity-check matrix H, action values  $Q_{0}(s_{a}^{(0)},a)$  corresponding to optimized global policy
Output: optimized local scheduling policy  $\hat{\pi}_k(s_{a_i}^{(I)})$
2  $\mathcal{B}_k\gets \emptyset ,Q^{(k)}(s_a^{(0)},a)\gets Q_0(s_a^{(0)},a)\forall s_a^{(0)},a$  // start of an episode
3 for each new  $\mathbf{L}'\in \mathcal{L}_k'$  do
4 perform Steps 7-27 of Algorithm 3 // adapt to the current SNR with index k
5 end
6  $Q^{(k)}(s_a^{(\ell)},a)\gets Q_0(s_a^{(\ell)},a)\forall s_a^{(\ell)},a$
7  $\pi_k(s_a^{(\ell)})\gets \arg \max_aQ^{(k)}(s_a^{(\ell)},a) / / k$  -th local policy update

The output of Algorithm 5 is an optimized CN scheduling policy,  $\hat{\pi}_k(s_{a_i}^{(I)})$ , for a new SNR, given by

$$
\hat {\pi} _ {k} \left(s _ {a _ {i}} ^ {(I)}\right) = \underset {a _ {i} \in \left[ \left[ \lceil m / z \rceil \right] \right] \backslash \left\{a _ {0}, \dots , a _ {i - 1} \right\}} {\arg \max } Q ^ {(k)} \left(s _ {a _ {i}} ^ {(I)}, a _ {i}\right). \tag {18}
$$

This policy is then used to select the CN updates during decoding in the same way as in the RELDEC scheme.

The iterative learning of the global policy along with the  $K$  local policies using Algorithms 3 and 4 enables fast adaptation to any new current channel condition through Algorithm 5 using only a relatively small number of LLR vectors in  $\mathcal{L}_k^{\prime}$ . This is in sharp contrast to the meta Q-learning scheme of [38], where a local policy is learned after learning the global policy, but the learned local policies are not employed for global

Authorized licensed use limited to: INTERNATIONAL INSTITUTE OF INFORMATION TECHNOLOGY. Downloaded on April 10,2026 at 19:55:22 UTC from IEEE Xplore. Restrictions apply.