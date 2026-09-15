function deg_groups = build_deg_groups(CN_neighbors)
% build_deg_groups - group check nodes by degree so their neighbor-LLR
% gather can be vectorized across all check nodes of the same degree
% instead of looped one check node at a time. Used with
% quartile_state_batch.m to make the quartile-average-LLR RELDEC
% architecture (docs/notes/quartile_llr_reldec.md) fast enough to run
% full eval sweeps in Octave.
%
% Returns a struct array, one entry per distinct check-node degree found
% in CN_neighbors, with fields:
%   .rows - check-node indices (into 1:m) having this degree
%   .idx  - (numel(rows) x degree) matrix of neighbor VN indices; row i
%           is CN_neighbors{rows(i)}
%
% For the WRAN matrix (row degree 10 or 11) this yields exactly 2 groups,
% turning what would be m=128 per-step function calls into 2 vectorized
% gather+sort operations.

degs = cellfun(@numel, CN_neighbors);
udeg = unique(degs);
deg_groups = struct('rows', {}, 'idx', {});
for u = 1:numel(udeg)
    d = udeg(u);
    rows = find(degs == d);
    idx = zeros(numel(rows), d);
    for r = 1:numel(rows)
        idx(r,:) = CN_neighbors{rows(r)};
    end
    deg_groups(end+1) = struct('rows', rows, 'idx', idx);
end
end
