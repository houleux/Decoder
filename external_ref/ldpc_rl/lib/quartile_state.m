function s = quartile_state(L, CN_neighbors, i, k)
% quartile_state - state encoder for the quartile-average-LLR RELDEC
% architecture (see docs/notes/quartile_llr_reldec.md).
%
% Sorts check node i's neighbor LLRs ascending, splits them into k
% groups as evenly as possible, and returns the average LLR within each
% group as a 1xk row vector.
%
% If the check node's degree is below k, only the first `deg` groups are
% non-empty; the trailing (k - deg) entries are returned as 0
% (zero-padded), mirroring how train/reldec_test.m zero-pads state slots
% beyond a check node's degree. For the WRAN matrix (row degree 10-11)
% with k in {3,4,5} this padding path is never exercised, but it keeps
% the encoder well-defined for lower-degree matrices.
%
% Shared by train/reldec_quartile_llr.m and eval/test_reldec_quartile_llr.m
% — keep both in sync with this file rather than re-deriving the state
% independently in each script.

idx  = CN_neighbors{i};
vals = sort(L(idx));
deg  = numel(vals);
kk   = min(deg, k);

edges = round(linspace(0, deg, kk+1));
s = zeros(1, k);
for g = 1:kk
    grp  = vals(edges(g)+1:edges(g+1));
    s(g) = mean(grp);
end
end
