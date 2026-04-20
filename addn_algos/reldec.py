"""
AM-RELDEC: Agile Meta-RELDEC for Sequential LDPC Decoding
==========================================================
Implementation of the RELDEC and AM-RELDEC algorithms from:

  Habib, Beemer, Kliewer — "RELDEC: Reinforcement Learning-Based Decoding of
  Moderate Length LDPC Codes", IEEE Trans. Commun., vol. 71, no. 10, Oct. 2023.
  DOI: 10.1109/TCOMM.2023.3296621

Only z = 1 (single-CN clusters) is supported, which achieves the best decoding
performance as reported in the paper.

Architecture overview
---------------------
- Each check node is its own cluster (z = 1).
- The MDP state for cluster `a` (i.e. check node `a`) is the integer formed by
  hard-deciding the posterior LLRs of the VNs adjacent to CN `a`.
  State space size = 2^{l_a}, where l_a = degree of CN `a`.
- The Q-table for cluster `a` is a dict  q[a][s_a] -> float  (sparse; only
  states that have been visited are stored).
- The Q-function value Q(s_a, a) represents the expected long-term reward for
  scheduling cluster `a` when its state is s_a.

Usage
-----
>>> import numpy as np
>>> from ldpc.bp_decoder import BpDecoder
>>> from addn_algos.am_reldec import AmReldec
>>>
>>> # H: binary parity-check matrix (m x n), clusters: list of lists of CN indices (z=1 → singletons)
>>> decoder = BpDecoder(H)
>>> reldec = AmReldec(H, clusters, decoder)
>>>
>>> # --- RELDEC training (single SNR) ---
>>> reldec.train_reldec(llr_list, codeword_list, snr_db=4.0)
>>>
>>> # --- Inference ---
>>> bits = reldec.decode(llr_vector, I_max=30)
>>>
>>> # --- AM-RELDEC meta-training (multiple SNRs) ---
>>> reldec.train_am_reldec(llr_dict, codeword_dict)   # llr_dict: {snr: [llr_vecs]}
>>>
>>> # --- Online adaptation from pilots ---
>>> reldec.adapt(pilot_llrs, pilot_codewords, target_snr)
"""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Helper: Q-table
# ---------------------------------------------------------------------------

class QTable:
    """Sparse Q-table for a single cluster.

    Stores Q(s, a) for all (state, global_action) pairs that have been
    visited.  Unvisited entries default to 0.0.

    Parameters
    ----------
    num_actions : int
        Total number of clusters / MDP actions.
    """

    def __init__(self, num_actions: int):
        self._num_actions = num_actions
        # dict[state_int -> np.ndarray of shape (num_actions,)]
        self._table: Dict[int, np.ndarray] = {}

    def get(self, state: int) -> np.ndarray:
        """Return Q(state, *) as a length-num_actions array (zeros if unseen)."""
        if state not in self._table:
            return np.zeros(self._num_actions, dtype=np.float64)
        return self._table[state].copy()

    def get_value(self, state: int, action: int) -> float:
        """Return Q(state, action)."""
        if state not in self._table:
            return 0.0
        return float(self._table[state][action])

    def set_value(self, state: int, action: int, value: float) -> None:
        """Set Q(state, action) = value."""
        if state not in self._table:
            self._table[state] = np.zeros(self._num_actions, dtype=np.float64)
        self._table[state][action] = value

    def copy(self) -> "QTable":
        """Return a deep copy."""
        qt = QTable(self._num_actions)
        for s, arr in self._table.items():
            qt._table[s] = arr.copy()
        return qt

    def average_with(self, others: List["QTable"]) -> None:
        """Replace self with the element-wise average of self + others (in-place).

        Used in Algorithm 4, Step 1: Q_global = (1/K) Σ_k Q_k.
        """
        all_tables = [self] + others
        K = len(all_tables)
        all_states: set = set()
        for t in all_tables:
            all_states.update(t._table.keys())
        new_table: Dict[int, np.ndarray] = {}
        for s in all_states:
            arrays = [t._table[s] if s in t._table
                      else np.zeros(self._num_actions, dtype=np.float64)
                      for t in all_tables]
            new_table[s] = np.mean(arrays, axis=0)
        self._table = new_table

    def update_from(self, other: "QTable") -> None:
        """Copy all entries from `other` into self (overwrite if present)."""
        for s, arr in other._table.items():
            self._table[s] = arr.copy()


# ---------------------------------------------------------------------------
# Core class
# ---------------------------------------------------------------------------

class Reldec:
    """RELDEC / AM-RELDEC decoder.

    Parameters
    ----------
    H : np.ndarray
        Binary parity-check matrix of shape (m, n).
    clusters : list of array-like
        Each element is a list/array of CN indices belonging to that cluster.
        For z = 1 this is simply [[0], [1], ..., [m-1]] or the caller-chosen
        grouping.  *Must* cover every CN exactly once.
    decoder : BpDecoder
        Initialised BpDecoder object wrapping the same H.  The caller owns
        this object; AmReldec calls reset(), initialise_log_domain_bp(),
        decode_cluster(), and log_prob_ratios on it.
    alpha : float
        Q-learning rate  (0 < α < 1).  Default 0.5.
    beta : float
        Reward discount factor  (0 < β < 1).  Default 0.9.
    epsilon : float
        Exploration probability for ε-greedy selection during training.
        Default 0.1.
    l_max : int
        Maximum Q-update steps per training episode.  Default = num_clusters.
    l_min_loss : float
        Loss threshold for early episode termination in AM-RELDEC.
        If the running average batch loss drops below this value the episode
        ends early.  Default 1e-4.
    x_steps : int
        Number of MDP instances between loss updates inside an episode
        (Algorithm 3, Step 20).  Default 5.
    """

    def __init__(
        self,
        H: np.ndarray,
        clusters: List,
        decoder,
        alpha: float = 0.5,
        beta: float = 0.9,
        epsilon: float = 0.1,
        l_max: Optional[int] = None,
        l_min_loss: float = 1e-4,
        x_steps: int = 5,
    ):
        self.H = np.array(H, dtype=np.int8)
        self.m, self.n = self.H.shape
        self.clusters: List[np.ndarray] = [np.asarray(c, dtype=np.int32) for c in clusters]
        self.num_clusters = len(self.clusters)

        # Enforce z = 1: every cluster must contain exactly one check node.
        bad = [i for i, c in enumerate(self.clusters) if len(c) != 1]
        if bad:
            raise ValueError(
                f"AmReldec only supports z=1 (single-CN clusters). "
                f"Clusters at indices {bad[:5]}{'...' if len(bad) > 5 else ''} "
                f"have sizes {[len(self.clusters[i]) for i in bad[:5]]}. "
                "Pass clusters=[[0],[1],...,[m-1]] for z=1."
            )
        self.decoder = decoder

        self.alpha = alpha
        self.beta = beta
        self.epsilon = epsilon
        self.l_max = l_max if l_max is not None else self.num_clusters
        self.l_min_loss = l_min_loss
        self.x_steps = x_steps

        # Build VN-adjacency for each cluster (which VN indices are neighbours)
        self.cluster_vns: List[np.ndarray] = self._build_cluster_vn_map()

        # Global Q-table (one per cluster)
        self.q_global: List[QTable] = [QTable(self.num_clusters)
                                        for _ in range(self.num_clusters)]

        # Local Q-tables keyed by SNR label (set after training)
        self.q_local: Dict[float, List[QTable]] = {}

        # Active policy: points to current global or adapted tables
        self.q_active: List[QTable] = self.q_global

    # ------------------------------------------------------------------
    # Graph utilities
    # ------------------------------------------------------------------

    def _build_cluster_vn_map(self) -> List[np.ndarray]:
        """For each cluster, find the union of VN indices adjacent to its CNs.

        Returns a list of length num_clusters, each element is a sorted
        np.ndarray of VN (column) indices.
        """
        cluster_vns = []
        for cluster in self.clusters:
            vn_set = set()
            for cn_idx in cluster:
                # iterate over non-zero columns in row cn_idx
                row = self.H[cn_idx]
                vns = np.where(row == 1)[0]
                vn_set.update(vns.tolist())
            cluster_vns.append(np.array(sorted(vn_set), dtype=np.int32))
        return cluster_vns

    # ------------------------------------------------------------------
    # State / reward
    # ------------------------------------------------------------------

    def _get_cluster_state(self, cluster_idx: int, log_prob_ratios: np.ndarray) -> int:
        """Compute integer state of cluster `cluster_idx`.

        Hard-decide the posterior LLRs of the cluster's neighbouring VNs,
        then convert the resulting binary vector to an integer (MSB first).
        Eq. (4) of the paper.

        Parameters
        ----------
        cluster_idx : int
        log_prob_ratios : np.ndarray  shape (n,)
            Current posterior LLRs from the decoder.

        Returns
        -------
        int in [0, 2^{l_a})
        """
        vns = self.cluster_vns[cluster_idx]
        bits = (log_prob_ratios[vns] < 0).astype(np.int32)  # 1 if LLR < 0
        # binary -> integer (MSB first)
        state = 0
        for b in bits:
            state = (state << 1) | int(b)
        return state

    def _get_reward(
        self,
        cluster_idx: int,
        x_transmitted: np.ndarray,
        log_prob_ratios: np.ndarray,
    ) -> float:
        """Compute reward R_a after scheduling cluster `cluster_idx`.

        Eq. (5):  R_a = (1/l_a) Σ 1(x_{j,a} == x̂_{j,a})

        Parameters
        ----------
        cluster_idx : int
        x_transmitted : np.ndarray  shape (n,)
            Full transmitted codeword (known during training).
        log_prob_ratios : np.ndarray  shape (n,)
            Current posterior LLRs.

        Returns
        -------
        float in [0, 1]
        """
        vns = self.cluster_vns[cluster_idx]
        if len(vns) == 0:
            return 0.0
        x_hat = (log_prob_ratios[vns] < 0).astype(np.int32)
        x_true = x_transmitted[vns].astype(np.int32)
        return float(np.mean(x_hat == x_true))

    # ------------------------------------------------------------------
    # ε-greedy action selection
    # ------------------------------------------------------------------

    def _select_action(
        self,
        states: List[int],
        excluded: set,
        q_tables: List[QTable],
        training: bool = True,
    ) -> int:
        """Select a cluster index using ε-greedy policy.

        Parameters
        ----------
        states : list of int
            Current state of every cluster (index = cluster idx).
        excluded : set of int
            Clusters already scheduled in this iteration (inference only).
        q_tables : list of QTable
        training : bool
            If True, apply ε-greedy; if False (inference), greedy only.

        Returns
        -------
        int  — chosen cluster index
        """
        available = [a for a in range(self.num_clusters) if a not in excluded]
        if not available:
            return -1

        if training and random.random() < self.epsilon:
            # Explore: uniform random
            return random.choice(available)

        # Exploit: argmax Q(s_a, a) over available actions
        best_val = -np.inf
        best_actions = []
        for a in available:
            val = q_tables[a].get_value(states[a], a)
            if val > best_val:
                best_val = val
                best_actions = [a]
            elif val == best_val:
                best_actions.append(a)
        return random.choice(best_actions)  # break ties uniformly

    # ------------------------------------------------------------------
    # Single-episode Q-learning step (shared by Alg 1, 3, 4, 5)
    # ------------------------------------------------------------------

    def _run_episode(
        self,
        llr_vector: np.ndarray,
        x_transmitted: Optional[np.ndarray],
        q_tables: List[QTable],
        collect_instances: bool = False,
        loss_threshold: float = np.inf,
    ) -> Tuple[float, List[Tuple]]:
        """Run one Q-learning episode on a single LLR vector.

        Parameters
        ----------
        llr_vector : np.ndarray  shape (n,)
        x_transmitted : np.ndarray or None
            If None, reward cannot be computed (inference-only mode — not used here).
        q_tables : list of QTable
            The Q-tables to update in-place.
        collect_instances : bool
            If True, accumulate (s, a, R, s') tuples and return them.
        loss_threshold : float
            Early-stop if running loss falls below this (AM-RELDEC).

        Returns
        -------
        (episode_loss, mdp_instances)
        """
        # --- Initialise decoder ---
        self.decoder.reset()
        self.decoder.initialise_log_domain_bp(llr_vector)
        current_llrs = llr_vector.copy()

        # --- Initial states (Eq. 4) ---
        states = [self._get_cluster_state(a, current_llrs)
                  for a in range(self.num_clusters)]

        mdp_instances: List[Tuple] = []
        batch_loss = 0.0
        instance_count = 0

        for step in range(self.l_max):
            # Select action ε-greedy
            a = self._select_action(states, set(), q_tables, training=True)
            if a < 0:
                break

            s_a = states[a]

            # Schedule cluster a: decode cluster-induced subgraph
            current_llrs = self.decoder.decode_cluster(
                self.clusters[a].tolist()
            )

            # New state of cluster a after scheduling
            s_a_prime = self._get_cluster_state(a, current_llrs)

            # Reward (Eq. 5)
            R_a = self._get_reward(a, x_transmitted, current_llrs) \
                if x_transmitted is not None else 0.0

            # Target value U(s_a, a) = R_a + β * max_{a'} Q(s_a', a')
            q_vec_prime = q_tables[a].get(s_a_prime)
            U = R_a + self.beta * float(np.max(q_vec_prime))

            # Q-update (Eq. 6)
            old_q = q_tables[a].get_value(s_a, a)
            new_q = (1 - self.alpha) * old_q + self.alpha * U
            q_tables[a].set_value(s_a, a, new_q)

            # TD error squared for loss computation
            td_err = U - old_q
            batch_loss += td_err * td_err
            instance_count += 1

            # Update state
            states[a] = s_a_prime

            if collect_instances:
                mdp_instances.append((s_a, a, R_a, s_a_prime))

            # Loss check every x_steps (Algorithm 3, Step 20)
            if collect_instances and instance_count > 0 and instance_count % self.x_steps == 0:
                running_loss = batch_loss / instance_count
                if running_loss < loss_threshold:
                    break  # early termination for AM-RELDEC

        episode_loss = batch_loss / instance_count if instance_count > 0 else 0.0
        return episode_loss, mdp_instances

    # ------------------------------------------------------------------
    # Algorithm 1: RELDEC Training
    # ------------------------------------------------------------------

    def train_reldec(
        self,
        llr_list: List[np.ndarray],
        codeword_list: List[np.ndarray],
        reset_qtables: bool = True,
    ) -> None:
        """Train RELDEC on a set of LLR / codeword pairs (Algorithm 1).

        Learns optimised CN scheduling policy for a single SNR (or SNR mix).

        Parameters
        ----------
        llr_list : list of np.ndarray  shape (n,)
            Training LLR vectors.  One per transmitted codeword.
        codeword_list : list of np.ndarray  shape (n,)
            Corresponding transmitted codewords (binary).
        reset_qtables : bool
            If True, reset Q-tables to zero before training.
        """
        if reset_qtables:
            self.q_global = [QTable(self.num_clusters)
                             for _ in range(self.num_clusters)]

        for llr, x in zip(llr_list, codeword_list):
            self._run_episode(
                llr_vector=np.asarray(llr, dtype=np.float64),
                x_transmitted=np.asarray(x, dtype=np.int32),
                q_tables=self.q_global,
                collect_instances=False,
            )

        self.q_active = self.q_global

    # ------------------------------------------------------------------
    # Algorithm 2: Inference (Learning-Based Sequential BP Decoding)
    # ------------------------------------------------------------------

    def decode(
        self,
        llr_vector: np.ndarray,
        I_max: int = 30,
        q_tables: Optional[List[QTable]] = None,
    ) -> np.ndarray:
        """Decode using the optimised CN scheduling policy (Algorithm 2).

        Parameters
        ----------
        llr_vector : np.ndarray  shape (n,)
            Channel LLR vector.
        I_max : int
            Maximum number of full decoder iterations.
        q_tables : list of QTable or None
            Policy to use.  Defaults to self.q_active (set after training).

        Returns
        -------
        np.ndarray  shape (n,)  uint8  — hard-decoded bits
        """
        if q_tables is None:
            q_tables = self.q_active

        self.decoder.reset()
        self.decoder.initialise_log_domain_bp(llr_vector)
        current_llrs = np.asarray(llr_vector, dtype=np.float64).copy()

        for _ in range(I_max):
            # For each slot in the iteration, schedule all clusters in order
            # determined by the learned policy (Eq. 10 / Alg. 2)
            scheduled: set = set()

            for _slot in range(self.num_clusters):
                # Determine states of unscheduled clusters
                states = [self._get_cluster_state(a, current_llrs)
                          for a in range(self.num_clusters)]

                # Select next cluster (greedy, no exploration)
                a_i = self._select_action(
                    states, scheduled, q_tables, training=False
                )
                if a_i < 0:
                    break
                scheduled.add(a_i)

                # Decode the chosen cluster
                current_llrs = self.decoder.decode_cluster(
                    self.clusters[a_i].tolist()
                )

            # Hard decisions
            decoded_bits = (current_llrs < 0).astype(np.uint8)

            # Syndrome check (stopping condition)
            syndrome = (self.H @ decoded_bits.astype(np.int32)) % 2
            if np.all(syndrome == 0):
                break

        return decoded_bits

    # ------------------------------------------------------------------
    # Algorithm 3 local phase + Algorithm 4 global phase: AM-RELDEC
    # ------------------------------------------------------------------

    def _local_phase(
        self,
        snr_key: float,
        llr_list: List[np.ndarray],
        codeword_list: List[np.ndarray],
        q_global_tables: List[QTable],
    ) -> List[QTable]:
        """Algorithm 3, Steps 4–31: adapt to k-th SNR.

        Initialises local Q from global, then learns on the k-th SNR dataset
        with early termination based on loss threshold.

        Returns the adapted local Q-tables.
        """
        # Step 5: initialise local from global
        q_local_k = [qt.copy() for qt in q_global_tables]

        batch_loss_total = 0.0
        num_instances = 0

        for llr, x in zip(llr_list, codeword_list):
            ep_loss, instances = self._run_episode(
                llr_vector=np.asarray(llr, dtype=np.float64),
                x_transmitted=np.asarray(x, dtype=np.int32),
                q_tables=q_local_k,
                collect_instances=True,
                loss_threshold=self.l_min_loss,
            )
            batch_loss_total += ep_loss * len(instances)
            num_instances += len(instances)

            # Update global reference (Step 29: Q_0 ← Q_ℓ for all states)
            for a in range(self.num_clusters):
                q_global_tables[a].update_from(q_local_k[a])

        return q_local_k

    def _global_phase(
        self,
        llr_list: List[np.ndarray],
        codeword_list: List[np.ndarray],
        q_local_all: List[List[QTable]],
        q_global_tables: List[QTable],
    ) -> None:
        """Algorithm 4: learn global policy as average of local then fine-tune.

        Modifies q_global_tables in-place.
        """
        # Step 1: Q_global = (1/K) Σ_k Q_k
        if q_local_all:
            for a in range(self.num_clusters):
                others = [q_local_all[k][a] for k in range(1, len(q_local_all))]
                q_global_tables[a].update_from(q_local_all[0][a])
                q_global_tables[a].average_with(others)

        # Steps 2-13: fine-tune global on the mixture dataset
        for llr, x in zip(llr_list, codeword_list):
            self._run_episode(
                llr_vector=np.asarray(llr, dtype=np.float64),
                x_transmitted=np.asarray(x, dtype=np.int32),
                q_tables=q_global_tables,
                collect_instances=True,
            )

    def train_am_reldec(
        self,
        llr_dict: Dict[float, List[np.ndarray]],
        codeword_dict: Dict[float, List[np.ndarray]],
        meta_iterations: int = 5,
        reset_qtables: bool = True,
    ) -> None:
        """Train AM-RELDEC (Algorithms 3 + 4, iterated).

        Parameters
        ----------
        llr_dict : dict {snr_db: [llr_vectors]}
            Training LLR vectors grouped by SNR.
        codeword_dict : dict {snr_db: [codewords]}
            Corresponding transmitted codewords grouped by SNR.
        meta_iterations : int
            Number of outer meta-learning rounds (outer while-loop, Alg. 3).
        reset_qtables : bool
            If True, reset global Q-tables before meta-training.
        """
        snr_keys = sorted(llr_dict.keys())
        K = len(snr_keys)

        if reset_qtables:
            self.q_global = [QTable(self.num_clusters)
                             for _ in range(self.num_clusters)]

        # Build global mixture  (all LLRs / codewords interleaved)
        all_llrs: List[np.ndarray] = []
        all_cwds: List[np.ndarray] = []
        for snr in snr_keys:
            all_llrs.extend(llr_dict[snr])
            all_cwds.extend(codeword_dict[snr])

        for meta_iter in range(meta_iterations):
            q_local_all: List[List[QTable]] = []

            # Algorithm 3, Steps 4-31: local phase for each SNR
            for snr in snr_keys:
                q_local_k = self._local_phase(
                    snr_key=snr,
                    llr_list=llr_dict[snr],
                    codeword_list=codeword_dict[snr],
                    q_global_tables=self.q_global,
                )
                q_local_all.append(q_local_k)
                self.q_local[snr] = q_local_k

            # Algorithm 4 (Step 32 of Alg. 3): global policy update
            self._global_phase(
                llr_list=all_llrs,
                codeword_list=all_cwds,
                q_local_all=q_local_all,
                q_global_tables=self.q_global,
            )

        self.q_active = self.q_global

    # ------------------------------------------------------------------
    # Algorithm 5: Online Adaptation (AM-RELDEC)
    # ------------------------------------------------------------------

    def adapt(
        self,
        pilot_llrs: List[np.ndarray],
        pilot_codewords: List[np.ndarray],
        target_snr: Optional[float] = None,
    ) -> List[QTable]:
        """Online adaptation to a new/current SNR from pilot signals (Alg. 5).

        Parameters
        ----------
        pilot_llrs : list of np.ndarray
            LLR vectors obtained from pilot symbols at the new SNR.
        pilot_codewords : list of np.ndarray
            Corresponding known pilot codewords.
        target_snr : float or None
            SNR label for caching the adapted policy.

        Returns
        -------
        list of QTable  — local (adapted) Q-tables.
        """
        # Step 1: initialise local from global (Alg. 5, Line 2)
        q_adapted = [qt.copy() for qt in self.q_global]

        # Steps 2-4: adapt using pilot data (Steps 7-27 of Alg. 3)
        for llr, x in zip(pilot_llrs, pilot_codewords):
            self._run_episode(
                llr_vector=np.asarray(llr, dtype=np.float64),
                x_transmitted=np.asarray(x, dtype=np.int32),
                q_tables=q_adapted,
                collect_instances=True,
                loss_threshold=self.l_min_loss,
            )

        # Cache adapted tables
        if target_snr is not None:
            self.q_local[target_snr] = q_adapted

        # Make this the active policy for subsequent decode() calls
        self.q_active = q_adapted
        return q_adapted

    # ------------------------------------------------------------------
    # Convenience: decode with a specific SNR's local policy
    # ------------------------------------------------------------------

    def decode_with_local(
        self,
        llr_vector: np.ndarray,
        snr_db: float,
        I_max: int = 30,
    ) -> np.ndarray:
        """Decode using the cached local policy for `snr_db` (if it exists).

        Falls back to the global policy if no local policy has been learned
        for this SNR.

        Parameters
        ----------
        llr_vector : np.ndarray  shape (n,)
        snr_db : float
        I_max : int

        Returns
        -------
        np.ndarray  shape (n,)  uint8
        """
        q_tables = self.q_local.get(snr_db, self.q_global)
        return self.decode(llr_vector, I_max=I_max, q_tables=q_tables)
