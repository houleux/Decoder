function stats = round_robin_eval_shard(shard, env)
% ROUND_ROBIN_EVAL_SHARD - evaluate ONE independent shard of frames for the
% cluster-level round-robin BP scheduler, and return accumulated statistics.
%
% DEFINITION (per explicit user direction, distinct from Python's
% rl/decoder/sequential.py::RoundRobinDecoder, which schedules individual
% check nodes and takes no z at all): a "cluster" is a contiguous group of z
% check nodes (env.clusters, built by lib/build_clusters.m -- the same
% grouping convention as rl/agents/reldec.py::ReldecAgent). Round-robin here
% means: schedule cluster 1, cluster 2, ..., cluster numClusters, in that
% fixed increasing order, every sweep -- no Q-table, no learning, no
% selection choice at all. env.i_max sweeps means each cluster is scheduled
% exactly env.i_max times total (one pass = env.i_max=1), with the syndrome
% checked once per completed pass and decoding stopping the moment it's zero.
% Scheduling a cluster means running the exact-BP CN->VN update for every
% check node in that cluster, in increasing index order within it (same
% convention as ReldecAgent processing agent.clusters[k]).
%
% z=1 reduces this to scheduling check nodes 1..m in order every pass, which
% IS equivalent to Python's RoundRobinDecoder -- a useful sanity check, but
% z=4 (and any z>1) has no Python counterpart to compare against; this whole
% file exists because the user asked for this specific MATLAB-only baseline.
%
% Pure function: no file I/O, no global state, no RNG side effects on the
% caller -- safe inside parfor, testable serially on its own.
%
% RNG / FRAME UNIQUENESS: same Threefry-substream scheme as
% quartile_eval_shard.m. Every shard gets a distinct, non-overlapping
% substream.
%
% INPUTS
%   shard - struct: .substream, .snr_db, .n_frames
%   env   - struct: .H, .CN_neighbors, .clusters, .m, .n, .code_rate, .i_max,
%           .base_seed
%
% OUTPUT
%   stats - struct: .frames .bit_errors .total_bits .frame_errors .messages
%           .converged_frames .wall_s -- same contract as quartile_eval_shard.m

H            = env.H;
CN_neighbors = env.CN_neighbors;
clusters     = env.clusters;
n            = env.n;
i_max        = env.i_max;
numClusters  = numel(clusters);

sigma = sqrt(1/(2*env.code_rate*10^(shard.snr_db/10)));

st = RandStream('Threefry', 'Seed', env.base_seed);
st.Substream = shard.substream;

bit_errors       = 0;
frame_errors     = 0;
messages         = 0;
converged_frames = 0;

t0 = tic;
for f = 1:shard.n_frames
    rx = 1 + sigma*randn(st, 1, n);
    L  = 2*rx/(sigma^2);

    m = numel(CN_neighbors);
    res = cell(m,1);
    for i = 1:m
        res{i} = zeros(1, numel(CN_neighbors{i}));
    end

    converged = false;
    for iter_idx = 1:i_max
        % ---- fixed round-robin order: cluster 1, 2, ..., numClusters ----
        for kk = 1:numClusters
            for cn = clusters{kk}
                idx2   = CN_neighbors{cn};
                vals2  = L(idx2) - res{cn};
                temp   = tanh(vals2./2);
                prodLq = prod(temp);
                res{cn} = 2*atanh(prodLq ./ temp);
                L(idx2) = vals2 + res{cn};
                messages = messages + numel(idx2);
            end
        end

        % ---- syndrome-based early stop (once per completed pass) ----
        hard = double(L < 0)';
        if all(mod(H*hard, 2) == 0)
            converged = true;
            break;
        end
    end

    hard_final   = (L < 0);
    nerr         = nnz(hard_final);          % transmitted codeword is all-zero
    bit_errors   = bit_errors + nerr;
    frame_errors = frame_errors + double(nerr > 0);
    converged_frames = converged_frames + double(converged);
end

stats = struct( ...
    'frames',           shard.n_frames, ...
    'bit_errors',       bit_errors, ...
    'total_bits',       shard.n_frames * n, ...
    'frame_errors',     frame_errors, ...
    'messages',         messages, ...
    'converged_frames', converged_frames, ...
    'wall_s',           toc(t0));
end
