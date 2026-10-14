function ckpt = sched_train(cfg)
% SCHED_TRAIN - train one cluster-scheduling policy and save a self-describing
% checkpoint. Same training protocol as this repo's Python run_experiments.py:
% one freshly drawn all-zero-codeword AWGN frame per episode, training SNRs
% cycled in order, cfg.l_max iterations per episode (see lib/sched_frame.m).
%
% cfg fields (all required unless a default is shown)
%   code           'wran' | 'bg2_z10'            z   cluster size
%   method         'reldec' | 'tanh_mi' | 'ppo'
%   episodes, l_max, train_snrs (vector), seed
%   out            checkpoint path (.mat); training is skipped if it exists
%   tabular:  alpha, gamma, epsilon
%   ppo:      ppo struct with lr, gamma, gae_lambda, clip_eps, update_epochs,
%             minibatch_size, rollout_steps, ent_coef, vf_coef, max_grad_norm,
%             hidden, reward ('success' | 'errors_success'), pos_feature
%   threads   (1) maxNumCompThreads for this MATLAB process
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
    case 'ppo'
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
    if mod(ep, max(100, round(cfg.episodes / 20))) == 0
        el = toc(t0);
        fprintf('  episode %d/%d  mean reward (last 100) %.4f  %.0fs elapsed, ~%.0fs left\n', ...
            ep, cfg.episodes, mean(recent(1:min(ep, 100))), el, el / ep * (cfg.episodes - ep));
    end
end
wall = toc(t0);

meta = rmfield(cfg, 'out');
meta.train_wall_s = wall;
meta.interpreter = version;
meta.source = 'external_ref/ldpc_rl/train/sched_train.m';
if strcmp(cfg.method, 'ppo'), meta.ppo_updates = updates; end
tmp = [cfg.out '.tmp.mat'];
pol_mem = pol;  pol = sched_policy_pack(pol);          % containers.Map -> plain arrays
save(tmp, 'pol', 'meta', '-v7');   % -v7: readable by both MATLAB and Octave
pol = pol_mem;   % -v7: readable by both MATLAB and Octave
movefile(tmp, cfg.out, 'f');
fprintf('sched_train: done in %.1fs -> %s\n', wall, cfg.out);
ckpt = struct('pol', pol, 'meta', meta);
end
