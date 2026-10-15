function ckpt = sched_train(cfg)
% SCHED_TRAIN - train one cluster-scheduling policy and save a self-describing
% checkpoint. Same training protocol as this repo's Python run_experiments.py:
% one freshly drawn all-zero-codeword AWGN frame per episode, training SNRs
% cycled in order, cfg.l_max iterations per episode (see lib/sched_frame.m).
%
% cfg fields (all required unless a default is shown)
%   code           'wran' | 'bg2_z10'            z   cluster size
%   method         'reldec' | 'tanh_mi' | 'ppo' | 'fast_ppo'
%   episodes, l_max, train_snrs (vector), seed
%   out            checkpoint path (.mat); training is skipped if it exists
%   tabular:  alpha, gamma, epsilon
%   ppo:      ppo struct with lr, gamma, gae_lambda, clip_eps, update_epochs,
%             minibatch_size, rollout_steps, ent_coef, vf_coef, max_grad_norm,
%             hidden, reward ('success' | 'errors_success'), pos_feature
%   fast_ppo: the ppo fields plus batch_frames (frames per batched rollout,
%             lib/fast_ppo_rollout.m) and collect_workers (parallel collectors
%             per round; > 1 opens a parpool of that size). Same network,
%             loss and update as ppo; collection is batched, noise/sampling come
%             from per-(round, worker) substreams (lib/fast_ppo_collect.m), and
%             training stops after the first round reaching cfg.episodes, so
%             meta.episodes (and snapshot episode counts) can exceed the target
%             by up to one round. The saved policy has kind 'ppo'.
%   threads   (1) maxNumCompThreads for this MATLAB process
%   snapshots, snapshot_outs  (none) also save the policy after episode
%             snapshots(i) to snapshot_outs{i}. Saving draws no random numbers and
%             PPO has no schedule tied to cfg.episodes, so a snapshot is identical
%             to a separate run of snapshots(i) episodes with the same seed.
%
% Reproducible: noise comes from RandStream('Threefry', Seed = seed) and every
% other draw (exploration, tie-breaks, sampling, shuffling, initial weights)
% from the global stream reseeded to seed + 7919.

here = fileparts(mfilename('fullpath'));
addpath(fullfile(here, '..', 'lib'));

req = {'code', 'z', 'method', 'episodes', 'l_max', 'train_snrs', 'seed', 'out'};
for i = 1:numel(req)
    if ~isfield(cfg, req{i}), error('sched_train:missing', 'cfg.%s is required', req{i}); end
end
if exist(cfg.out, 'file')
    fprintf('sched_train: %s exists, skipping training\n', cfg.out);
    ckpt = load(cfg.out);
    ckpt.pol = sched_policy_unpack(ckpt.pol);
    return;
end
isOct = exist('OCTAVE_VERSION', 'builtin') ~= 0;
if ~isfield(cfg, 'threads'), cfg.threads = 1; end
if ~isOct, maxNumCompThreads(cfg.threads); end

env = sched_env(cfg.code, cfg.z);
n = env.n;  K = env.K;
if isOct
    randn('state', cfg.seed);  rand('state', cfg.seed + 7919);
    gen = @() randn(1, n);
else
    st = RandStream('Threefry', 'Seed', cfg.seed);
    RandStream.setGlobalStream(RandStream('Threefry', 'Seed', cfg.seed + 7919));
    gen = @() randn(st, 1, n);
end

updates = [];
switch cfg.method
    case {'reldec', 'tanh_mi'}
        pol = struct('kind', cfg.method, 'alpha', cfg.alpha, 'gamma', cfg.gamma, 'epsilon', cfg.epsilon);
        if strcmp(cfg.method, 'tanh_mi')
            pol.dense = true;  pol.Q = zeros(21, K);
        else
            pol.pow2 = 2 .^ (0:env.maxNb-1)';
            pol.dense = env.maxNb <= 20;
            if pol.dense
                pol.store = 'dense';  pol.Q = zeros(2^env.maxNb, K);
            elseif env.maxNb <= 52                 % sparse numeric keys: unseen states have Q = 0
                pol.store = 'keys';
                pol.keys = zeros(1024, K);  pol.vals = zeros(1024, K);  pol.cnt = zeros(1, K);
            else                                   % > 52 bits: a double can't hold the key exactly
                pol.store = 'map';  pol.maps = cell(1, K);
                for k = 1:K, pol.maps{k} = containers.Map('KeyType', 'char', 'ValueType', 'double'); end
            end
        end
    case {'ppo', 'fast_ppo'}
        hp = cfg.ppo;
        D = 6 + double(hp.pos_feature) + 3;
        pol = struct('kind', 'ppo', 'net', ppo_init(D, hp.hidden), ...
                     'pos_feature', logical(hp.pos_feature), 'reward', hp.reward);
        cap = hp.rollout_steps + K * cfg.l_max;
        buf = struct('X', zeros(K, D, cap, 'single'), 'mask', false(K, cap), 'a', zeros(1, cap), ...
                     'logp', zeros(1, cap), 'v', zeros(1, cap), 'r', zeros(1, cap), 'done', false(1, cap));
        N = 0;  adam = [];  updates = 0;
    otherwise
        error('sched_train:badMethod', 'unknown method "%s"', cfg.method);
end

opt = struct('i_max', cfg.l_max, 'train', true);
fprintf('sched_train: %s on %s z=%d (%d clusters), %d episodes, l_max=%d, SNRs %s, seed %d\n', ...
    cfg.method, cfg.code, cfg.z, K, cfg.episodes, cfg.l_max, mat2str(cfg.train_snrs), cfg.seed);
t0 = tic;  recent = zeros(1, 100);  nS = numel(cfg.train_snrs);
if strcmp(cfg.method, 'fast_ppo')
    [pol, updates, eps_done, tim] = fast_ppo_train(cfg, env, pol, hp, isOct, t0);
    wall = toc(t0);
    save_ckpt(cfg.out, pol, cfg, eps_done, wall, updates, tim);
    fprintf('sched_train: done in %.1fs (collect %.0fs, update %.0fs) -> %s\n', wall, tim.collect_s, tim.update_s, cfg.out);
    ckpt = struct('pol', pol, 'meta', make_meta(cfg, eps_done, wall, updates, tim));
    return;
end
for ep = 1:cfg.episodes
    snr = cfg.train_snrs(mod(ep - 1, nS) + 1);
    sigma = sqrt(1 / (2 * env.code_rate * 10^(snr / 10)));
    Lch = 2 * (1 + sigma * gen()) / sigma^2;
    [o, pol] = sched_frame(Lch, env, pol, opt);
    recent(mod(ep - 1, 100) + 1) = o.reward;
    if strcmp(cfg.method, 'ppo')
        T = numel(o.ep.a);
        buf.X(:, :, N+1:N+T) = o.ep.X;  buf.mask(:, N+1:N+T) = o.ep.mask;
        buf.a(N+1:N+T) = o.ep.a;  buf.logp(N+1:N+T) = o.ep.logp;  buf.v(N+1:N+T) = o.ep.v;
        buf.r(N+1:N+T) = o.ep.r;  buf.done(N+1:N+T) = o.ep.done;
        N = N + T;
        if N >= hp.rollout_steps
            [pol.net, adam] = ppo_update(pol.net, adam, buf, N, hp);
            N = 0;  updates = updates + 1;
        end
    end
    if isfield(cfg, 'snapshots') && any(cfg.snapshots == ep)
        save_ckpt(cfg.snapshot_outs{cfg.snapshots == ep}, pol, cfg, ep, toc(t0), updates);
    end
    if mod(ep, max(100, round(cfg.episodes / 20))) == 0
        el = toc(t0);
        fprintf('  episode %d/%d  mean reward (last 100) %.4f  %.0fs elapsed, ~%.0fs left\n', ...
            ep, cfg.episodes, mean(recent(1:min(ep, 100))), el, el / ep * (cfg.episodes - ep));
    end
end
wall = toc(t0);
save_ckpt(cfg.out, pol, cfg, cfg.episodes, wall, updates);
fprintf('sched_train: done in %.1fs -> %s\n', wall, cfg.out);
ckpt = struct('pol', pol, 'meta', make_meta(cfg, cfg.episodes, wall, updates));
end


function [pol, updates, eps_done, tim] = fast_ppo_train(cfg, env, pol, hp, isOct, t0)
W = hp.collect_workers;  quota = ceil(hp.rollout_steps / W);
if W > 1 && ~isOct && isempty(gcp('nocreate'))
    c = parcluster('local');  jsl = tempname;  mkdir(jsl);  % private job storage (see sched_train_batch.m)
    c.JobStorageLocation = jsl;  c.NumWorkers = W;
    parpool(c, W);
end
snaps = [];  outs = {};
if isfield(cfg, 'snapshots'), snaps = cfg.snapshots(:)';  outs = cfg.snapshot_outs; end
saved = false(size(snaps));
updates = 0;  eps_done = 0;  adam = [];  round_ = 0;
tim = struct('collect_s', 0, 'update_s', 0);
while eps_done < cfg.episodes
    round_ = round_ + 1;  net = pol.net;  parts = cell(1, W);
    tc = tic;
    if W > 1
        parfor w = 1:W
            parts{w} = fast_ppo_collect(env, net, pol.pos_feature, cfg.l_max, cfg.train_snrs, quota, ...
                                        hp.batch_frames, cfg.seed, (round_ - 1) * W + w);
        end
    else
        parts{1} = fast_ppo_collect(env, net, pol.pos_feature, cfg.l_max, cfg.train_snrs, quota, ...
                                    hp.batch_frames, cfg.seed, round_);
    end
    tim.collect_s = tim.collect_s + toc(tc);
    X = cell(1, W);  for w = 1:W, X{w} = parts{w}.ep.X; end
    buf.X = cat(3, X{:});
    for f = {'mask', 'a', 'logp', 'v', 'r', 'done'}
        Y = cell(1, W);  for w = 1:W, Y{w} = parts{w}.ep.(f{1}); end
        buf.(f{1}) = [Y{:}];
    end
    tu = tic;
    [pol.net, adam] = ppo_update(pol.net, adam, buf, numel(buf.a), hp);
    tim.update_s = tim.update_s + toc(tu);
    updates = updates + 1;
    E = sum(cellfun(@(p) p.episodes, parts));  eps_done = eps_done + E;
    for i = find(~saved & eps_done >= snaps)
        save_ckpt(outs{i}, pol, cfg, eps_done, toc(t0), updates, tim);  saved(i) = true;
    end
    if mod(round_, 50) == 0
        el = toc(t0);
        fprintf('  round %d: %d/%d episodes  success rate %.3f  %.0fs elapsed (collect %.0f, update %.0f), ~%.0fs left\n', ...
            round_, eps_done, cfg.episodes, sum(cellfun(@(p) p.reward, parts)) / E, el, tim.collect_s, ...
            tim.update_s, el / eps_done * (cfg.episodes - eps_done));
    end
end
end


function meta = make_meta(cfg, episodes, wall, updates, tim)
meta = rmfield(cfg, intersect(fieldnames(cfg), {'out', 'snapshots', 'snapshot_outs'}));
meta.episodes = episodes;
meta.train_wall_s = wall;
meta.interpreter = version;
meta.source = 'external_ref/ldpc_rl/train/sched_train.m';
if any(strcmp(cfg.method, {'ppo', 'fast_ppo'})), meta.ppo_updates = updates; end
if nargin > 4, meta.collect_s = tim.collect_s;  meta.update_s = tim.update_s; end
end


function save_ckpt(out, pol, cfg, episodes, wall, updates, varargin)
meta = make_meta(cfg, episodes, wall, updates, varargin{:});
pol = sched_policy_pack(pol);             % containers.Map -> plain arrays
tmp = [out '.tmp.mat'];
save(tmp, 'pol', 'meta', '-v7');          % -v7: readable by both MATLAB and Octave
movefile(tmp, out, 'f');
fprintf('sched_train: saved %d-episode policy -> %s\n', episodes, out);
end
