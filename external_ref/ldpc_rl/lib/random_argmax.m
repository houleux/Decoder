function idx = random_argmax(vals)
% RANDOM_ARGMAX - argmax with uniform-random tie-breaking.
%
% MATLAB's own max() breaks ties by returning the FIRST maximal index,
% deterministically. That is a real bug for RL action selection against a
% freshly zero-initialized Q-table (or any Q-table with genuine ties): every
% episode's first pick would deterministically be the lowest-indexed
% candidate until its Q-value moves off the pack, systematically starving
% higher-indexed clusters of early training signal.
%
% This repo's Python reference breaks ties uniformly at random --
% rl/agents/reldec.py::select_cluster:
%     best_q = max(q_values)
%     best = [k for k, q in zip(available_clusters, q_values) if q == best_q]
%     return rng.choice(best)
% This function reproduces that behaviour so the MATLAB replication's action
% selection matches Python's, not just its scheduling-loop structure.
%
% INPUT
%   vals - vector of Q-values for the candidate set (same order as caller's
%          index list; caller maps idx back to its own indexing)
% OUTPUT
%   idx  - index (into vals) of a uniformly-random maximal element

m = max(vals);
ties = find(vals == m);
if numel(ties) == 1
    idx = ties(1);
else
    idx = ties(randi(numel(ties)));
end
end
