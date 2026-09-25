function Q = RELDEC_QUARTILE_SWEEP_MAIN(L_set, H, CN_neighbors, deg_groups, clusters, params)
% RELDEC_QUARTILE_SWEEP_MAIN - tabular Q-learning trainer for the
% quartile-average-LLR RELDEC architecture, using the SAME scheduling and
% Q-update semantics as this repo's Python rl/trainer.py::train_episode() +
% rl/algorithms/factored_q_learning.py::FactoredQLearning.update().
%
% THIS IS A CORRECTED REPLACEMENT FOR lib/RELDEC_QUARTILE_MAIN.m, not an
% edit of it -- RELDEC_QUARTILE_MAIN.m stays exactly as recovered/committed,
% since it is what train/reldec_quartile_llr.m (the original, untouched
% reference driver) depends on and what produced the historical results in
% docs/notes/quartile_llr_reldec_results.md.
%
% WHY A NEW FILE: RELDEC_QUARTILE_MAIN.m reproduces reldec_test.m's original
% mechanics -- lmax RAW GREEDY PICKS with no sweep guarantee (a cluster can
% be picked repeatedly while others are never touched), bootstrapped against
% the GLOBAL MAX over every cluster's own next-state Q-value. Python's
% factored MDP design is different on both counts:
%   1. Scheduling: each l_max iteration is a full SWEEP -- every one of the m
%      clusters is scheduled EXACTLY ONCE (removed from an `available` set as
%      it's picked), with an early stop the moment the syndrome is all-zero,
%      checked once per completed sweep (not mid-sweep).
%   2. Q-update: bootstrapped ONLY against that same cluster's own next-state
%      Q-value -- no cross-cluster max. Each cluster's sub-MDP is genuinely
%      independent; comparison across clusters happens only at action-
%      selection time, never inside the value update.
% Using RELDEC_QUARTILE_MAIN.m's original mechanics to claim a comparison
% against Python's reldec would silently compare two different algorithms
% under the same label -- this file exists specifically to avoid that.
%
% Also uses lib/random_argmax.m for uniform-random tie-breaking (MATLAB's own
% max() deterministically returns the first maximal index, which would bias
% every episode's first pick against a freshly zero-initialized Q-table --
% Python's rl/agents/reldec.py::select_cluster breaks ties uniformly at
% random and this replication needs to match that, not just the loop shape).
%
% See external_ref/ldpc_rl/parallel_eval/quartile_eval_shard.m for the
% matching (epsilon=0, no Q-update) eval-time version of this same loop.

alpha   = params.alpha;
gamma   = params.beta;   % kept as params.beta for interface compatibility with
                          % the original RELDEC_QUARTILE_MAIN.m / reldec_test.m
                          % convention; semantically this is Python's `gamma`
                          % (discount factor), not a "beta" cross-cluster
                          % bootstrap weight -- there is no cross-cluster term
                          % in this version.
epsilon = params.epsilon;
lmax    = params.lmax;
k       = params.k;

m = length(CN_neighbors);
numClusters = length(clusters);
maxStates = 2^k;

Q = zeros(maxStates, numClusters);
N = length(L_set);
pow2vec = 2.^(k-1:-1:0);
cols = (1:m)';
tic;

for idx = 1:N

    L = L_set{idx}(:)';   % row
    res = cell(m,1);
    for i = 1:m
        res{i} = zeros(1, numel(CN_neighbors{i}));
    end

    S = quartile_state_batch(L, deg_groups, k, m);
    state_hard = S < 0;
    s = 1 + state_hard * pow2vec';
    vals1 = Q(sub2ind(size(Q), s, cols))';

    for l = 1:lmax
        available = true(1, numClusters);   % logical mask over 1:numClusters
        n_left = numClusters;

        while n_left > 0
            avail_idx = find(available);

            %% -------- ACTION (epsilon-greedy, random tie-break, among AVAILABLE only) --------
            if rand < epsilon
                a = avail_idx(randi(numel(avail_idx)));
            else
                local = random_argmax(vals1(avail_idx));
                a = avail_idx(local);
            end
            available(a) = false;
            n_left = n_left - 1;

            % state BEFORE this cluster's update (current live state, i.e.
            % reflecting every OTHER cluster already scheduled earlier in this
            % same sweep -- matches Python's llr_post being a single mutable
            % array read fresh by select_cluster() on every pick)
            s_before = s(a);

            %% -------- CN -> VN (exact BP), identical formula to reldec_test.m --------
            idx2 = CN_neighbors{a};
            vals2 = L(idx2) - res{a};
            temp = tanh(vals2./2);
            prodLq = prod(temp);
            res{a} = 2*atanh(prodLq ./ temp);
            L(idx2) = vals2 + res{a};

            %% -------- NEW STATE (recomputed for all m clusters; L is global) --------
            S_new = quartile_state_batch(L, deg_groups, k, m);
            state_hard_new = S_new < 0;

            %% -------- REWARD (reldec_test.m formula, k in place of maxStateBits) --------
            deg_a  = length(CN_neighbors{a});
            k_eff  = min(deg_a, k);
            reward = (nnz(state_hard_new(a,:) == 0) - (k - k_eff)) / k_eff;

            s_new_all = 1 + state_hard_new * pow2vec';
            s_after = s_new_all(a);

            %% -------- Q UPDATE: OWN CLUSTER'S NEXT STATE ONLY, no cross-cluster max --------
            q_before = Q(s_before, a);
            q_after  = Q(s_after, a);
            Q(s_before, a) = q_before + alpha * (reward + gamma * q_after - q_before);

            % refresh live Q-lookup for the remaining picks in this sweep (only
            % s/vals1 are read going forward; state_hard itself is not reused
            % until it's rebuilt fresh at the top of the next episode)
            s = s_new_all;
            vals1 = Q(sub2ind(size(Q), s, cols))';
        end

        %% -------- SYNDROME EARLY STOP (once per completed sweep, not mid-sweep) --------
        hard = double(L < 0)';
        if all(mod(H*hard, 2) == 0)
            break;
        end
    end

    if mod(idx, 100) == 0
        elapsed = toc;
        rate = idx / elapsed;
        remaining = (N - idx) / rate;
        fprintf('Episode %d/%d (%.2f%%) | ETA: %.1fs\n', idx, N, 100*idx/N, remaining);
    end
end
end
