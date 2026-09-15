function Q = RELDEC_QUARTILE_MAIN(L_set, CN_neighbors, deg_groups, clusters, params)
% RELDEC_QUARTILE_MAIN - tabular Q-learning trainer for the
% quartile-average-LLR RELDEC architecture. See
% train/reldec_quartile_llr.m for the caller and
% docs/notes/quartile_llr_reldec.md for design rationale. Split into its
% own function file (rather than a script-trailing local function, as
% reldec_test.m and friends use) for portability: Octave does not support
% MATLAB's script-local-function form used by the upstream reference
% scripts.
%
% Uses quartile_state_batch.m (with deg_groups from build_deg_groups.m,
% precomputed once by the caller) instead of calling quartile_state.m once
% per check node per step — verified numerically identical, but ~2 vectorized
% gather+sort ops instead of m=128 per step, which is what makes running
% this in Octave (no JIT) tractable.

alpha   = params.alpha;
beta    = params.beta;
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
    S = quartile_state_batch(L, deg_groups, k, m);
    state_hard = S < 0;
    res = cell(m,1);
    for i = 1:m
        res{i} = zeros(1, numel(CN_neighbors{i}));
    end

    s = 1 + state_hard * pow2vec';
    vals1 = Q(sub2ind(size(Q), s, cols))';

    % start of an episode
    for l = 1:lmax
        %% -------- ACTION --------
        if rand < epsilon
            a = randi(numClusters);
        else
            [~, a] = max(vals1);
        end
        %% -------- CN -> VN (exact BP) --------
        idx2 = CN_neighbors{a};
        vals2 = L(idx2) - res{a};
        temp = tanh(vals2./2);
        prodLq = prod(temp);
        res{a} = 2*atanh(prodLq ./ temp);
        L(idx2) = vals2 + res{a};

        %% -------- NEW STATE --------
        S_updated = quartile_state_batch(L, deg_groups, k, m);
        state_hard_updated = S_updated < 0;

        %% -------- REWARD (reldec_test.m formula, k in place of maxStateBits) --------
        deg_a  = length(CN_neighbors{a});
        k_eff  = min(deg_a, k);   % non-padded quantile groups for this cluster
        reward = (nnz(state_hard_updated(a,:) == 0) - (k - k_eff)) / k_eff;

        s_new = 1 + state_hard_updated * pow2vec';
        vals1 = Q(sub2ind(size(Q), s_new, cols))';

        %% -------- Q UPDATE --------
        Q(s(a),a) = (1-alpha)*Q(s(a),a) + alpha*(reward + beta*max(vals1));

        % State update for next iteration
        s = s_new;
    end

    %% -------- PROGRESS --------
    if mod(idx,1000) == 0
        elapsed = toc;
        rate = idx / elapsed;
        remaining = (N - idx) / rate;
        fprintf('Episode %d/%d (%.2f%%) | ETA: %.1fs\n', ...
            idx, N, 100*idx/N, remaining);
    end
end
end
