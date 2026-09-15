function S = quartile_state_batch(L, deg_groups, k, m)
% quartile_state_batch - vectorized form of quartile_state.m: computes the
% k-element quartile-average-LLR state for ALL m check nodes in one call
% (one gather+sort per distinct check-node degree, via deg_groups from
% build_deg_groups.m) instead of m individual calls.
%
% Must produce the exact same result as calling quartile_state(L,
% CN_neighbors, i, k) for every i — see docs/notes/quartile_llr_reldec.md.
% If a check node's degree d < k, only the first d columns of that row
% are filled and the rest are left at 0 (zero-padded), matching
% quartile_state.m.
%
% L         - 1xn channel/working LLR vector
% deg_groups- struct array from build_deg_groups(CN_neighbors)
% k         - number of quantile groups
% m         - total number of check nodes (rows of the output)

S = zeros(m, k);
for gi = 1:numel(deg_groups)
    rows = deg_groups(gi).rows;
    idxmat = deg_groups(gi).idx;      % (numel(rows) x d)
    d = size(idxmat, 2);

    Lg = reshape(L(idxmat), size(idxmat));
    sorted = sort(Lg, 2);

    kk = min(d, k);
    edges = round(linspace(0, d, kk+1));
    grp = zeros(numel(rows), k);
    for g = 1:kk
        grp(:, g) = mean(sorted(:, edges(g)+1:edges(g+1)), 2);
    end
    S(rows, :) = grp;
end
end
