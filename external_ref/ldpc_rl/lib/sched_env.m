function env = sched_env(code, z)
% SCHED_ENV - graph, clusters and lookup tables for the cluster-scheduling
% decoders (sched_frame.m / flood_frame.m), for one code and cluster size.
%
% Mirrors this repo's Python setup (rl/agents/reldec.py::ReldecAgent.__init__):
% clusters are contiguous groups of z check nodes, and a cluster's
% neighborhood is the sorted union of its check nodes' variable nodes.
%
% INPUTS
%   code - 'wran'     matrices/wran_384_256.mat (128x384; byte-identical to the
%                     repo's matrices/WRAN_irreg_384_256.csv)
%          'bg2_z10'  matrices/P_520.mat expanded with block size 10 (420x520;
%                     identical to the repo's matrices/H_BG2_Z10.csv, the
%                     [520,100] 5G-NR code of the RELDEC paper)
%   z    - cluster size (check nodes per cluster)
%
% OUTPUT env fields
%   H (sparse double), m, n, code_rate, nnzH
%   CN_neighbors{c}  ascending VN indices of check c
%   res0             cell of zero CN->VN message vectors (copy to start a frame)
%   clusters{k}, K, clMsgs(k) (messages per scheduling of cluster k)
%   nb{k}            sorted unique VN neighborhood of cluster k; nbLen(k), maxNb
%   NB               K x maxNb VN indices, padded with n+1 (callers keep
%                    L(n+1) = 0, so pads read as tanh 0 / "not > 0")
%   padmask          NB == n+1
%   Cavg             K x m sparse: row k averages over cluster k's checks
%   pos              (0:K-1)'/(K-1)
%   er, ec           edge list (row, col) for the flooding decoder
%   matrix_repo, matrix_matlab   paths recorded in manifests

here = fileparts(mfilename('fullpath'));
matdir = fullfile(here, '..', 'matrices');
switch code
    case 'wran'
        d = load(fullfile(matdir, 'wran_384_256.mat'), 'wran_384_256');
        H = sparse(double(logical(d.wran_384_256)));
        env.matrix_repo   = 'matrices/WRAN_irreg_384_256.csv';
        env.matrix_matlab = 'matrices/wran_384_256.mat';
    case 'bg2_z10'
        d = load(fullfile(matdir, 'P_520.mat'), 'P_520');
        H = qc_expand(double(d.P_520), 10);
        env.matrix_repo   = 'matrices/H_BG2_Z10.csv';
        env.matrix_matlab = 'matrices/P_520.mat (block size 10)';
    otherwise
        error('sched_env:badCode', 'unknown code "%s" (expected wran or bg2_z10)', code);
end
[m, n] = size(H);
env.code = code;  env.z = z;
env.H = H;  env.m = m;  env.n = n;  env.code_rate = 1 - m/n;  env.nnzH = nnz(H);

CN_neighbors = cell(m, 1);  res0 = cell(m, 1);
for c = 1:m
    CN_neighbors{c} = find(H(c, :));
    res0{c} = zeros(1, numel(CN_neighbors{c}));
end
env.CN_neighbors = CN_neighbors;  env.res0 = res0;

clusters = build_clusters(m, z);
K = numel(clusters);
nb = cell(K, 1);  nbLen = zeros(K, 1);  clMsgs = zeros(K, 1);
ci = [];  cj = [];  cv = [];
for k = 1:K
    nb{k} = unique([CN_neighbors{clusters{k}}]);
    nbLen(k) = numel(nb{k});
    for c = clusters{k}
        clMsgs(k) = clMsgs(k) + numel(CN_neighbors{c});
    end
    ci = [ci, k*ones(1, numel(clusters{k}))];                    %#ok<AGROW>
    cj = [cj, clusters{k}];                                      %#ok<AGROW>
    cv = [cv, ones(1, numel(clusters{k}))/numel(clusters{k})];   %#ok<AGROW>
end
maxNb = max(nbLen);
NB = (n + 1) * ones(K, maxNb);
for k = 1:K
    NB(k, 1:nbLen(k)) = nb{k};
end
env.clusters = clusters;  env.K = K;  env.clMsgs = clMsgs;
env.nb = nb;  env.nbLen = nbLen;  env.maxNb = maxNb;
env.NB = NB;  env.padmask = (NB == n + 1);
env.Cavg = sparse(ci, cj, cv, K, m);
env.pos = (0:K-1)' / max(1, K - 1);
[er, ec] = find(H);
env.er = er;  env.ec = ec;
end

function H = qc_expand(P, Z)
% Quasi-cyclic expansion: entry p >= 0 becomes the Z x Z identity cyclically
% shifted right by p (row r has its one in column mod(r + p, Z), 0-based);
% entry -1 becomes the zero block. Same convention as MATLAB's
% ldpcQuasiCyclicMatrix, written out so no toolbox is needed.
[mb, nbk] = size(P);
r = [];  c = [];
for i = 1:mb
    for j = 1:nbk
        if P(i, j) >= 0
            r = [r, (i-1)*Z + (1:Z)];                           %#ok<AGROW>
            c = [c, (j-1)*Z + mod((0:Z-1) + P(i, j), Z) + 1];   %#ok<AGROW>
        end
    end
end
H = sparse(r, c, 1, mb*Z, nbk*Z);
end
