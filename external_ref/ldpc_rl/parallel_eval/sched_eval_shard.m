function stats = sched_eval_shard(shard, env, pol, i_max, base_seed)
% SCHED_EVAL_SHARD - evaluate ONE independent shard of frames for any of the
% cluster-scheduling decoders and return accumulated statistics.
%
% Pure function (no file I/O), safe inside parfor. Frames are all-zero
% codeword, BPSK + AWGN, drawn from a Threefry substream unique to this shard
% (same scheme as round_robin_eval_shard.m), so no two shards share a frame.
%
% INPUTS
%   shard     .substream .snr_db .n_frames
%   env       from lib/sched_env.m
%   pol       .kind = 'flooding' (lib/flood_frame.m) or a policy for
%             lib/sched_frame.m ('round_robin' | 'reldec' | 'tanh_mi' | 'ppo')
%   i_max     decoding iterations
%   base_seed sweep-wide seed
% OUTPUT stats: .frames .bit_errors .total_bits .frame_errors .messages
%               .converged_frames .iterations .wall_s

n = env.n;
sigma = sqrt(1 / (2 * env.code_rate * 10^(shard.snr_db / 10)));
if exist('OCTAVE_VERSION', 'builtin')
    % Octave has no RandStream: local testing only, NOT the substream guarantee.
    randn('state', base_seed * 100003 + shard.substream);
    draw = @() randn(1, n);
else
    st = RandStream('Threefry', 'Seed', base_seed);
    st.Substream = shard.substream;
    draw = @() randn(st, 1, n);
end
opt = struct('i_max', i_max, 'train', false);
flooding = strcmp(pol.kind, 'flooding');

bit_errors = 0;  frame_errors = 0;  messages = 0;  converged_frames = 0;  iterations = 0;
t0 = tic;
if strcmp(pol.kind, 'ppo')
    % Batched greedy decoding (lib/fast_ppo_rollout.m; lib/fast_ppo_smoketest.m
    % checks it against sched_frame.m). Noise is drawn frame by frame exactly as
    % below, so the shard's frames are unchanged.
    Lall = zeros(shard.n_frames, n);
    for f = 1:shard.n_frames, Lall(f, :) = 2 * (1 + sigma * draw()) / sigma^2; end
    for s = 1:250:shard.n_frames
        o = fast_ppo_rollout(Lall(s:min(s + 249, shard.n_frames), :), env, pol.net, pol.pos_feature, i_max, false, []);
        nerr = sum(o.L < 0, 2);
        bit_errors = bit_errors + sum(nerr);  frame_errors = frame_errors + nnz(nerr);
        messages = messages + sum(o.messages);  iterations = iterations + sum(o.iterations);
        converged_frames = converged_frames + nnz(o.converged);
    end
    stats = struct('frames', shard.n_frames, 'bit_errors', bit_errors, 'total_bits', shard.n_frames * n, ...
                   'frame_errors', frame_errors, 'messages', messages, ...
                   'converged_frames', converged_frames, 'iterations', iterations, 'wall_s', toc(t0));
    return;
end
for f = 1:shard.n_frames
    rx  = 1 + sigma * draw();
    Lch = 2 * rx / sigma^2;
    if flooding
        o = flood_frame(Lch, env, i_max);
        nerr = nnz(o.L <= 0);            % transmitted codeword is all-zero
    else
        o = sched_frame(Lch, env, pol, opt);
        nerr = nnz(o.L < 0);
    end
    bit_errors   = bit_errors + nerr;
    frame_errors = frame_errors + double(nerr > 0);
    messages     = messages + o.messages;
    iterations   = iterations + o.iterations;
    converged_frames = converged_frames + double(o.converged);
end
stats = struct('frames', shard.n_frames, 'bit_errors', bit_errors, 'total_bits', shard.n_frames * n, ...
               'frame_errors', frame_errors, 'messages', messages, ...
               'converged_frames', converged_frames, 'iterations', iterations, 'wall_s', toc(t0));
end
