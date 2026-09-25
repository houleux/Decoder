function stats = quartile_eval_shard(shard, env)
% QUARTILE_EVAL_SHARD - evaluate ONE independent shard of frames for the
% quartile-average-LLR RELDEC policy, and return accumulated statistics.
%
% This is the numerical core of the parallel eval. It is a PURE function: no
% file I/O, no global state, no RNG side effects on the caller. That makes it
% (a) safe to call from inside parfor, and (b) testable serially on its own.
% All persistence is the caller's job (see run_quartile_eval.m).
%
% SCHEDULING SEMANTICS (matches Python's rl/agents/reldec.py::decode())
%   Each of env.maxIter iterations is a full SWEEP: every one of the m
%   clusters is scheduled EXACTLY ONCE (greedy, epsilon=0, uniform-random tie
%   break via random_argmax.m), removed from an `available` set as it's
%   picked. Syndrome is checked once per COMPLETED sweep, not mid-sweep, and
%   decoding stops the moment it hits zero.
%
%   An earlier version of this file used lmax RAW GREEDY PICKS with no sweep
%   guarantee (the same mechanics as external_ref/ldpc_rl/train/reldec_test.m)
%   -- found to not be a fair comparison against Python's reldec, which always
%   schedules every cluster once per iteration. That earlier run's results
%   were deleted from experiments.db and archived to
%   docs/notes/archived_results/ rather than silently overwritten with no
%   record. This version replaces it.
%
% RNG / FRAME UNIQUENESS
%   Frames are drawn from an explicitly positioned counter-based random stream:
%       st = RandStream('Threefry', 'Seed', env.base_seed);
%       st.Substream = shard.substream;
%   Threefry substreams are guaranteed non-overlapping, and shard.substream is
%   unique across every (snr, chunk) pair in the whole sweep. Two shards
%   therefore cannot generate the same frame -- including across different SNR
%   points.
%
% INPUTS
%   shard - struct:
%       .substream  unique positive integer (global across the sweep)
%       .snr_db     SNR in dB for this shard
%       .n_frames   number of frames to simulate in this shard
%   env   - struct (read-only, broadcast to all workers):
%       .Q            2^k x m Q-table (trained policy)
%       .k            number of quantile groups
%       .CN_neighbors m x 1 cell of neighbor VN indices
%       .deg_groups   from build_deg_groups()
%       .H            sparse parity-check matrix (m x n)
%       .m, .n        dimensions
%       .code_rate    R
%       .maxIter      scheduling budget in SWEEPS (m*maxIter actions/frame,
%                     worst case; fewer if the syndrome converges early)
%       .base_seed    sweep-wide RNG seed
%
% OUTPUT
%   stats - struct with fields matching expdb's commit_chunk() contract:
%       .frames .bit_errors .total_bits .frame_errors .messages
%   plus .converged_frames and .wall_s for diagnostics.
%
%   .messages counts CN->VN message updates (degree of each scheduled check
%   node), matching rl/decoder/engine.py's MethodStats.messages convention so
%   the MATLAB and Python rows in expdb mean the same thing.

Q            = env.Q;
k            = env.k;
CN_neighbors = env.CN_neighbors;
deg_groups   = env.deg_groups;
H            = env.H;
m            = env.m;
n            = env.n;
maxIter      = env.maxIter;

pow2vec = 2.^(k-1:-1:0);
cols    = (1:m)';
sigma   = sqrt(1/(2*env.code_rate*10^(shard.snr_db/10)));

% --- independent, non-overlapping random stream for this shard ---
st = RandStream('Threefry', 'Seed', env.base_seed);
st.Substream = shard.substream;

bit_errors       = 0;
frame_errors     = 0;
messages         = 0;
converged_frames = 0;

t0 = tic;
for f = 1:shard.n_frames
    % all-zero codeword, BPSK + AWGN -> channel LLRs
    rx = 1 + sigma*randn(st, 1, n);
    L  = 2*rx/(sigma^2);

    res = cell(m,1);
    for i = 1:m
        res{i} = zeros(1, numel(CN_neighbors{i}));
    end

    S = quartile_state_batch(L, deg_groups, k, m);
    state_hard = S < 0;
    s = 1 + state_hard * pow2vec';
    vals1 = Q(sub2ind(size(Q), s, cols))';

    converged = false;

    for iter_idx = 1:maxIter
        available = true(1, m);
        n_left = m;

        while n_left > 0
            avail_idx = find(available);

            % ---- greedy action (epsilon = 0 at eval), random tie-break, among AVAILABLE only ----
            local = random_argmax(vals1(avail_idx));
            a = avail_idx(local);
            available(a) = false;
            n_left = n_left - 1;

            % ---- CN -> VN (exact BP) ----
            idx2   = CN_neighbors{a};
            vals2  = L(idx2) - res{a};
            temp   = tanh(vals2./2);
            prodLq = prod(temp);
            res{a} = 2*atanh(prodLq ./ temp);
            L(idx2) = vals2 + res{a};
            messages = messages + numel(idx2);

            % ---- new state (recomputed for all m clusters; L is global) ----
            S = quartile_state_batch(L, deg_groups, k, m);
            state_hard = S < 0;
            s = 1 + state_hard * pow2vec';
            vals1 = Q(sub2ind(size(Q), s, cols))';
        end

        % ---- syndrome-based early stop (once per completed sweep) ----
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
