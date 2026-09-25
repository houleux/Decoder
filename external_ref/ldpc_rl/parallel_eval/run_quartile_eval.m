function summary = run_quartile_eval(cfg)
% RUN_QUARTILE_EVAL - parallel, resumable, incrementally-persisted BER/FER
% sweep for the quartile-average-LLR RELDEC policy.
%
% Splits the sweep into independent (snr, chunk) shards, runs them across a
% parpool, and writes each finished chunk to a spool directory the moment it
% completes. A separate single-writer Python process (matlab_bridge.ingest)
% moves those chunks into DuckDB, so results accumulate in the database WHILE
% the sweep runs and an abrupt termination loses at most one in-flight chunk
% per worker.
%
% The numerical decode loop lives in quartile_eval_shard.m and is identical to
% eval/test_reldec_quartile_llr.m. That original script is NOT modified.
%
% FRAME UNIQUENESS ACROSS WORKERS
%   Every shard gets a distinct Threefry substream index, assigned from a
%   single global counter over the whole shard list. Threefry substreams are
%   guaranteed non-overlapping, so no two shards -- on any worker, at any SNR
%   -- can ever draw the same frame. parfor's default RNG behaviour does not
%   guarantee this, and silently duplicated frames would inflate apparent
%   precision without changing the BER point estimate, which is exactly the
%   kind of error that is invisible in the output.
%
% CONFIG FIELDS (defaults in parens)
%   checkpoint   (REQUIRED)  path to Q_*.mat written by reldec_quartile_train
%   snr_db       (1:6)       SNR points in dB
%   n_frames     (100000)    frames per SNR point
%   chunk_size   (250)       frames per spool chunk (crash granularity vs overhead)
%   maxIter      (5)         scheduling budget in sweeps; m*maxIter actions/frame
%   base_seed    (12345)     sweep-wide RNG seed; substreams derive from it
%   n_workers    (36)        parpool size
%   spool_dir    (REQUIRED)  where chunk files are written
%   target_frame_errors (n_frames)  stored in expdb's eval key; equal to
%                            n_frames means "no early stop" (see note below)
%   method_name  ('matlab_reldec_quartile_k<k>')  expdb method name; MUST keep
%                            the 'matlab_' prefix so MATLAB-produced rows are
%                            distinguishable from this repo's Python runs
%   matfile/matvar (from checkpoint meta)
%
% NOTE ON target_frame_errors
%   The prior 30000-episode run had FER = 1.0 at every SNR in 0-4 dB. A
%   conventional target_frame_errors of ~100 would therefore stop each SNR
%   point after ~100 frames and silently defeat the requested frame budget.
%   Defaulting it to n_frames means only the frame budget governs completion.
%
% OUTPUT
%   summary - struct array, one entry per SNR, with aggregate BER/FER. Note the
%             authoritative record is DuckDB (via the ingester); this return
%             value is a convenience for interactive use.

if nargin < 1, cfg = struct(); end

here = fileparts(mfilename('fullpath'));
addpath(fullfile(here, '..', 'lib'));
addpath(here);

% ---------------- defaults ----------------
def = struct('snr_db', 1:6, 'n_frames', 100000, 'chunk_size', 250, ...
             'maxIter', 5, 'base_seed', 12345, 'n_workers', 36, ...
             'target_frame_errors', [], 'method_name', '', ...
             'matfile', '', 'matvar', '');
fn = fieldnames(def);
for i = 1:numel(fn)
    if ~isfield(cfg, fn{i}), cfg.(fn{i}) = def.(fn{i}); end
end
if ~isfield(cfg, 'checkpoint') || isempty(cfg.checkpoint)
    error('run_quartile_eval:noCheckpoint', 'cfg.checkpoint is required (no default).');
end
if ~isfield(cfg, 'spool_dir') || isempty(cfg.spool_dir)
    error('run_quartile_eval:noSpoolDir', 'cfg.spool_dir is required (no default).');
end
if isempty(cfg.target_frame_errors)
    cfg.target_frame_errors = cfg.n_frames;   % no early stop
end

%% ---------------- LOAD TRAINED POLICY ----------------
d = load(cfg.checkpoint, 'Q', 'params', 'meta');
Q = d.Q;
k = d.params.k;
meta = d.meta;
if ~isfield(meta, 'mechanics')
    error('run_quartile_eval:noMechanics', ...
        ['checkpoint %s has no meta.mechanics field, so we cannot tell whether ' ...
         'it was trained with sweep or raw_pick scheduling semantics. Retrain ' ...
         'with reldec_quartile_train.m (which always records this) rather than ' ...
         'guessing.'], cfg.checkpoint);
end
if isempty(cfg.matfile), cfg.matfile = meta.matfile; end
if isempty(cfg.matvar),  cfg.matvar  = meta.matvar;  end
if isempty(cfg.method_name)
    cfg.method_name = sprintf('matlab_reldec_quartile_k%d', k);
end
if ~startsWith(cfg.method_name, 'matlab_')
    error('run_quartile_eval:badMethodName', ...
        ['method_name must start with "matlab_" so MATLAB-produced rows stay ' ...
         'distinguishable from this repo''s Python runs (got "%s")'], cfg.method_name);
end

%% ---------------- MATRIX / GRAPH ----------------
mf = cfg.matfile;
if ~(startsWith(mf, '/'))
    mf = fullfile(here, '..', 'train', mf);   % meta paths are relative to train/
end
dm = load(mf, cfg.matvar);
H = sparse(logical(dm.(cfg.matvar)));
[m, n] = size(H);
R = 1 - m/n;

CN_neighbors = cell(m,1);
for c = 1:m
    CN_neighbors{c} = find(H(c,:));
end
deg_groups = build_deg_groups(CN_neighbors);

env = struct('Q', Q, 'k', k, 'CN_neighbors', {CN_neighbors}, ...
             'deg_groups', deg_groups, 'H', H, 'm', m, 'n', n, ...
             'code_rate', R, 'maxIter', cfg.maxIter, 'base_seed', cfg.base_seed);

%% ---------------- BUILD SHARD LIST ----------------
shards = struct('substream', {}, 'snr_db', {}, 'n_frames', {}, 'chunk_id', {});
sub = 0;
for si = 1:numel(cfg.snr_db)
    remaining = cfg.n_frames;
    ci = 0;
    while remaining > 0
        nf = min(cfg.chunk_size, remaining);
        sub = sub + 1;
        shards(end+1) = struct( ...
            'substream', sub, ...
            'snr_db',    cfg.snr_db(si), ...
            'n_frames',  nf, ...
            'chunk_id',  sprintf('k%d_snr%g_c%06d', k, cfg.snr_db(si), ci)); %#ok<AGROW>
        remaining = remaining - nf;
        ci = ci + 1;
    end
end
nShards = numel(shards);

fprintf('run_quartile_eval: k=%d  method=%s\n', k, cfg.method_name);
fprintf('  matrix %dx%d rate %.4f | SNR %s dB | %d frames/SNR | chunk %d\n', ...
    m, n, R, mat2str(cfg.snr_db), cfg.n_frames, cfg.chunk_size);
fprintf('  %d shards, budget %d actions/frame, base_seed %d\n', ...
    nShards, m*cfg.maxIter, cfg.base_seed);

%% ---------------- MANIFEST (config for the ingester) ----------------
if ~exist(cfg.spool_dir, 'dir'), mkdir(cfg.spool_dir); end
manifest = struct( ...
    'method',              cfg.method_name, ...
    'matrix',              'matrices/WRAN_irreg_384_256.csv', ... % byte-identical to the .mat; use the repo-side path so MATLAB and Python rows are comparable
    'matrix_matlab',       cfg.matfile, ...
    'z',                   1, ...              % one cluster == one check node
    'k',                   k, ...
    'state_encoding',      'quartile_avg_llr', ...
    'scheduling',          meta.mechanics, ...  % 'sweep': matches Python rl/trainer.py; part of config identity so it can never collide with a 'raw_pick' run under the same config_id
    'alpha',               meta.alpha, ...
    'gamma',               meta.beta, ...
    'epsilon',             meta.epsilon, ...
    'l_max',               meta.lmax, ...
    'train_episodes',      meta.numSamples, ...
    'train_snr_vals',      meta.snr_db, ...
    'seed',                meta.seed, ...
    'eval_maxIter',        cfg.maxIter, ...
    'eval_base_seed',      cfg.base_seed, ...
    'target_frame_errors', cfg.target_frame_errors, ...
    'max_frames',          cfg.n_frames, ...
    'checkpoint',          cfg.checkpoint, ...
    'n_shards',            nShards, ...
    'chunk_size',          cfg.chunk_size, ...
    'interpreter',         version, ...
    'source',              'external_ref/ldpc_rl/parallel_eval/run_quartile_eval.m');
mfid = fopen(fullfile(cfg.spool_dir, 'manifest.json'), 'w');
fprintf(mfid, '%s', jsonencode(manifest));
fclose(mfid);
fprintf('  manifest -> %s\n', fullfile(cfg.spool_dir, 'manifest.json'));

%% ---------------- PARPOOL ----------------
p = gcp('nocreate');
if isempty(p) || p.NumWorkers ~= cfg.n_workers
    if ~isempty(p), delete(p); end
    c = parcluster('local');
    if c.NumWorkers < cfg.n_workers
        c.NumWorkers = cfg.n_workers;   % local profile defaults below core count
        saveProfile(c);
    end
    p = parpool(c, cfg.n_workers);
end
fprintf('  parpool: %d workers\n', p.NumWorkers);

%% ---------------- RUN ----------------
spool_dir  = cfg.spool_dir;
tfe        = cfg.target_frame_errors;
maxf       = cfg.n_frames;
methodname = cfg.method_name;

t0 = tic;
results = cell(nShards, 1);
parfor si = 1:nShards
    sh = shards(si);
    st = quartile_eval_shard(sh, env);

    payload = struct( ...
        'method',              methodname, ...
        'snr_db',              sh.snr_db, ...
        'target_frame_errors', tfe, ...
        'max_frames',          maxf, ...
        'frames',              st.frames, ...
        'bit_errors',          st.bit_errors, ...
        'total_bits',          st.total_bits, ...
        'frame_errors',        st.frame_errors, ...
        'messages',            st.messages, ...
        'converged_frames',    st.converged_frames, ...
        'substream',           sh.substream, ...
        'wall_s',              st.wall_s);

    spool_write_chunk(spool_dir, sh.chunk_id, payload);
    results{si} = st;
    fprintf('  [shard %d/%d] snr=%g frames=%d ber=%.4g fer=%.4g (%.1fs)\n', ...
        si, nShards, sh.snr_db, st.frames, ...
        st.bit_errors/st.total_bits, st.frame_errors/st.frames, st.wall_s);
end
wall = toc(t0);

%% ---------------- SUMMARY ----------------
summary = struct('snr_db', {}, 'frames', {}, 'ber', {}, 'fer', {}, 'converged', {});
for si = 1:numel(cfg.snr_db)
    snr = cfg.snr_db(si);
    fr = 0; be = 0; tb = 0; fe = 0; cv = 0;
    for j = 1:nShards
        if shards(j).snr_db == snr
            fr = fr + results{j}.frames;      be = be + results{j}.bit_errors;
            tb = tb + results{j}.total_bits;  fe = fe + results{j}.frame_errors;
            cv = cv + results{j}.converged_frames;
        end
    end
    summary(end+1) = struct('snr_db', snr, 'frames', fr, 'ber', be/tb, ...
                            'fer', fe/fr, 'converged', cv); %#ok<AGROW>
end

fprintf('\n===== SWEEP COMPLETE (%.1fs wall, %.4g s/frame effective) =====\n', ...
    wall, wall/(numel(cfg.snr_db)*cfg.n_frames));
fprintf('%8s %10s %14s %12s %12s\n', 'SNR_dB', 'frames', 'BER', 'FER', 'converged');
for si = 1:numel(summary)
    fprintf('%8g %10d %14.6g %12.6g %12d\n', summary(si).snr_db, summary(si).frames, ...
        summary(si).ber, summary(si).fer, summary(si).converged);
end
end
