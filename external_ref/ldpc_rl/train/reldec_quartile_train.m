function outfiles = reldec_quartile_train(cfg)
% RELDEC_QUARTILE_TRAIN - parameterized trainer for the quartile-average-LLR
% RELDEC architecture.
%
% This is the reusable/function form of train/reldec_quartile_llr.m (which is a
% script with k = [3 4 5] and numSamples = 30000 hardcoded). The numerical core
% calls lib/RELDEC_QUARTILE_SWEEP_MAIN.m by default (see cfg.mechanics below),
% which matches this repo's Python rl/trainer.py scheduling/bootstrap exactly.
% It also calls the same lib/RELDEC_QUARTILE_MAIN.m trainer, the same
% lib/quartile_state_batch.m encoder, and generates training LLRs with the same
% all-zero-codeword BPSK+AWGN generator. Only the configuration is lifted into
% arguments so the same file can serve any (k, episode count, SNR, seed) without
% editing it.
%
% The original script is left untouched -- this file does not modify or shadow
% it, and writes checkpoints under a distinct name (see cfg.tag) so existing
% checkpoints such as Q_wran_quartile_k3.mat are never overwritten.
%
% USAGE
%   cfg.k_vals     = 3;                 % scalar or vector of k values
%   cfg.numSamples = 500;               % training episodes per k
%   outfiles = reldec_quartile_train(cfg);
%
% CONFIG FIELDS (all optional except where noted; defaults in parens)
%   k_vals      (3)        quantile-group counts to train, scalar or vector
%   numSamples  (500)      training episodes per k
%   alpha       (0.1)      Q-learning rate
%   beta        (0.9)      discount factor
%   epsilon     (0.1)      exploration probability during training
%   lmax        (50)       scheduling steps per episode
%   snr_linear  (1)        training SNR, LINEAR (1 == 0 dB); matches
%                          reldec_quartile_llr.m, which trains at snr(1) = 1
%   seed        (42)       RNG seed -- set explicitly so training is reproducible
%                          (the original script never seeded, so its run is not)
%   matfile     ('../matrices/wran_384_256.mat')  parity-check matrix .mat
%   matvar      ('wran_384_256')                  variable name inside it
%   outdir      ('../checkpoints')                where checkpoints are written
%   tag         ('ep%d' with numSamples)  filename suffix, keeps runs distinct
%   mechanics   ('sweep')  'sweep' calls lib/RELDEC_QUARTILE_SWEEP_MAIN.m --
%                          full sweep per l_max iteration (every cluster
%                          scheduled exactly once), own-cluster-only Q
%                          bootstrap, syndrome early-stop once per sweep --
%                          matching Python's rl/trainer.py exactly. 'raw_pick'
%                          calls the ORIGINAL lib/RELDEC_QUARTILE_MAIN.m
%                          (lmax raw greedy picks, no sweep guarantee, global-
%                          max bootstrap) -- kept only to reproduce the
%                          archived 2026-09-15 result in
%                          docs/notes/archived_results/, which used that
%                          mechanic and was found to not be a fair comparison
%                          against Python reldec. Default changed to 'sweep'
%                          for that reason.
%
% RETURNS
%   outfiles - cell array of written checkpoint paths, one per k
%
% Each checkpoint stores Q plus the full params struct (including seed and
% episode count) so a checkpoint is always self-describing -- eval and the
% expdb config builder both read the run configuration back out of it rather
% than having it re-specified by hand.

if nargin < 1, cfg = struct(); end

% ---------------- defaults (explicit; no silent fallbacks) ----------------
def = struct( ...
    'k_vals',     3, ...
    'numSamples', 500, ...
    'alpha',      0.1, ...
    'beta',       0.9, ...
    'epsilon',    0.1, ...
    'lmax',       50, ...
    'snr_linear', 1, ...
    'seed',       42, ...
    'matfile',    '../matrices/wran_384_256.mat', ...
    'matvar',     'wran_384_256', ...
    'outdir',     '../checkpoints', ...
    'tag',        '', ...
    'mechanics',  'sweep');
fn = fieldnames(def);
for i = 1:numel(fn)
    if ~isfield(cfg, fn{i}), cfg.(fn{i}) = def.(fn{i}); end
end
if isempty(cfg.tag)
    cfg.tag = sprintf('ep%d', cfg.numSamples);
end

% ---------------- resolve paths relative to THIS file --------------------
% so the function works regardless of the caller's cwd (the original scripts
% require being run from their own directory).
here = fileparts(mfilename('fullpath'));
addpath(fullfile(here, '..', 'lib'));
resolve = @(p) resolve_path(p, here);

%% ---------------- PARITY CHECK MATRIX ----------------
mf = resolve(cfg.matfile);
d = load(mf, cfg.matvar);
H = sparse(logical(d.(cfg.matvar)));
[m, n] = size(H);
R = 1 - m/n;   % code rate direct from H (reldec_test.m leaves R undefined)

fprintf('reldec_quartile_train: matrix %s  (%dx%d, rate %.4f)\n', mf, m, n, R);

%% ---------------- PRECOMPUTE GRAPH ----------------
CN_neighbors = cell(m,1);
for c = 1:m
    CN_neighbors{c} = find(H(c,:));
end
deg_groups = build_deg_groups(CN_neighbors);
clusters = num2cell(1:m);   % one cluster == one check node

%% ---------------- GENERATE TRAINING DATA ----------------
% Same all-zero-codeword BPSK+AWGN generator as reldec_quartile_llr.m, but
% seeded so the run is reproducible.
rng(cfg.seed, 'twister');
sigma = sqrt(1/(2*R*cfg.snr_linear));
L_set = cell(cfg.numSamples,1);
for i = 1:cfg.numSamples
    rx = 1 + sigma*randn(1,n);
    L_set{i} = 2*rx/(sigma^2);
end
fprintf('  generated %d training frames at snr_linear=%g (%.2f dB), sigma=%.6f, seed=%d\n', ...
    cfg.numSamples, cfg.snr_linear, 10*log10(cfg.snr_linear), sigma, cfg.seed);

%% ---------------- TRAIN FOR EACH k ----------------
outdir = resolve(cfg.outdir);
if ~exist(outdir, 'dir'), mkdir(outdir); end

outfiles = cell(1, numel(cfg.k_vals));
for kk = 1:numel(cfg.k_vals)
    params = struct('alpha', cfg.alpha, 'beta', cfg.beta, 'epsilon', cfg.epsilon, ...
                    'lmax', cfg.lmax, 'k', cfg.k_vals(kk));

    fprintf('=== Training quartile-LLR RELDEC, k = %d, episodes = %d, mechanics = %s ===\n', ...
        params.k, cfg.numSamples, cfg.mechanics);
    t0 = tic;
    switch cfg.mechanics
        case 'sweep'
            Q = RELDEC_QUARTILE_SWEEP_MAIN(L_set, H, CN_neighbors, deg_groups, clusters, params);
        case 'raw_pick'
            Q = RELDEC_QUARTILE_MAIN(L_set, CN_neighbors, deg_groups, clusters, params);
        otherwise
            error('reldec_quartile_train:badMechanics', ...
                'cfg.mechanics must be ''sweep'' or ''raw_pick'', got ''%s''', cfg.mechanics);
    end
    train_s = toc(t0);

    % Self-describing checkpoint: everything needed to reproduce / to build the
    % expdb config row travels with the Q-table.
    meta = struct('k', params.k, 'numSamples', cfg.numSamples, 'alpha', cfg.alpha, ...
                  'mechanics', cfg.mechanics, ...
                  'beta', cfg.beta, 'epsilon', cfg.epsilon, 'lmax', cfg.lmax, ...
                  'snr_linear', cfg.snr_linear, 'snr_db', 10*log10(cfg.snr_linear), ...
                  'seed', cfg.seed, 'matfile', cfg.matfile, 'matvar', cfg.matvar, ...
                  'm', m, 'n', n, 'code_rate', R, 'train_wall_s', train_s, ...
                  'trainer', 'reldec_quartile_train.m');

    outfile = fullfile(outdir, sprintf('Q_wran_quartile_k%d_%s.mat', params.k, cfg.tag));
    save(outfile, 'Q', 'params', 'meta', '-v7');
    outfiles{kk} = outfile;
    fprintf('  trained in %.2fs -> %s\n', train_s, outfile);
end
end

function p = resolve_path(p, here)
% Treat relative paths as relative to this file's directory, not the cwd.
if ~(startsWith(p, '/') || (ispc && numel(p) > 1 && p(2) == ':'))
    p = fullfile(here, p);
end
end
