function summary = run_round_robin_eval(cfg)
% RUN_ROUND_ROBIN_EVAL - parallel, resumable, incrementally-persisted BER/FER
% sweep for the cluster-level round-robin BP scheduler (see
% round_robin_eval_shard.m for the exact definition and why it differs from
% Python's rl/decoder/sequential.py::RoundRobinDecoder).
%
% Structurally the eval-only half of run_quartile_eval.m -- same shard/spool/
% manifest machinery, same ingester on the Python side -- but there is no
% checkpoint to load: round-robin has no Q-table and no training phase at
% all, so this orchestrator builds the graph/clusters directly from cfg
% rather than reading a trained-policy .mat file.
%
% CONFIG FIELDS (defaults in parens)
%   z            (REQUIRED)  cluster size -- contiguous groups of z check
%                nodes, built via lib/build_clusters.m. z=1 reduces to
%                Python's RoundRobinDecoder; z>1 has no Python counterpart.
%   snr_db       (1:6)       SNR points in dB
%   n_frames     (100000)    frames per SNR point
%   chunk_size   (250)       frames per spool chunk
%   i_max        (5)         sweeps; each cluster scheduled exactly i_max
%                times total (fewer if the syndrome converges early)
%   base_seed    (REQUIRED)  sweep-wide RNG seed; substreams derive from it.
%                No default -- see run_round_robin.m, which assigns a
%                DIFFERENT base_seed per z so that z=1 and z=4 (which are
%                otherwise fed the identical shard layout) draw independent
%                Monte Carlo samples rather than becoming a copy-paste of
%                each other's random stream.
%   n_workers    (36)        parpool size
%   spool_dir    (REQUIRED)  where chunk files are written
%   target_frame_errors (n_frames)  no early stop by default -- see the note
%                in run_quartile_eval.m for why (short of an explicit ask,
%                defaulting to n_frames is what avoids silently truncating
%                the requested frame budget at low FER)
%   method_name  ('matlab_round_robin')
%   matfile      ('../matrices/wran_384_256.mat')
%   matvar       ('wran_384_256')
%
% OUTPUT
%   summary - struct array, one entry per SNR, aggregate BER/FER. The
%             authoritative record is DuckDB via the ingester; this is a
%             convenience for interactive use.

if nargin < 1, cfg = struct(); end

here = fileparts(mfilename('fullpath'));
addpath(fullfile(here, '..', 'lib'));

def = struct('snr_db', 1:6, 'n_frames', 100000, 'chunk_size', 250, ...
             'i_max', 5, 'n_workers', 36, 'target_frame_errors', [], ...
             'method_name', 'matlab_round_robin', ...
             'matfile', '../matrices/wran_384_256.mat', 'matvar', 'wran_384_256');
fn = fieldnames(def);
for i = 1:numel(fn)
    if ~isfield(cfg, fn{i}), cfg.(fn{i}) = def.(fn{i}); end
end
if ~isfield(cfg, 'z') || isempty(cfg.z)
    error('run_round_robin_eval:noZ', 'cfg.z is required (no default).');
end
if ~isfield(cfg, 'base_seed') || isempty(cfg.base_seed)
    error('run_round_robin_eval:noBaseSeed', ...
        'cfg.base_seed is required (no default) -- see run_round_robin.m.');
end
if ~isfield(cfg, 'spool_dir') || isempty(cfg.spool_dir)
    error('run_round_robin_eval:noSpoolDir', 'cfg.spool_dir is required (no default).');
end
if isempty(cfg.target_frame_errors)
    cfg.target_frame_errors = cfg.n_frames;
end
if ~startsWith(cfg.method_name, 'matlab_')
    error('run_round_robin_eval:badMethodName', ...
        ['method_name must start with "matlab_" so MATLAB-produced rows stay ' ...
         'distinguishable from this repo''s Python runs (got "%s")'], cfg.method_name);
end

%% ---------------- MATRIX / GRAPH ----------------
mf = cfg.matfile;
if ~startsWith(mf, '/')
    mf = fullfile(here, mf);
end
dm = load(mf, cfg.matvar);
H = sparse(logical(dm.(cfg.matvar)));
[m, n] = size(H);
R = 1 - m/n;

CN_neighbors = cell(m,1);
for c = 1:m
    CN_neighbors{c} = find(H(c,:));
end
clusters = build_clusters(m, cfg.z);
numClusters = numel(clusters);

env = struct('H', H, 'CN_neighbors', {CN_neighbors}, 'clusters', {clusters}, ...
             'm', m, 'n', n, 'code_rate', R, 'i_max', cfg.i_max, ...
             'base_seed', cfg.base_seed);

fprintf('run_round_robin_eval: z=%d  method=%s\n', cfg.z, cfg.method_name);
fprintf('  matrix %dx%d rate %.4f | %d clusters | SNR %s dB | %d frames/SNR | chunk %d\n', ...
    m, n, R, numClusters, mat2str(cfg.snr_db), cfg.n_frames, cfg.chunk_size);

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
            'chunk_id',  sprintf('rr_z%d_snr%g_c%06d', cfg.z, cfg.snr_db(si), ci)); %#ok<AGROW>
        remaining = remaining - nf;
        ci = ci + 1;
    end
end
nShards = numel(shards);
fprintf('  %d shards, budget %d actions/frame, base_seed %d\n', ...
    nShards, numClusters*cfg.i_max, cfg.base_seed);

%% ---------------- MANIFEST (config for the ingester) ----------------
if ~exist(cfg.spool_dir, 'dir'), mkdir(cfg.spool_dir); end
manifest = struct( ...
    'method',              cfg.method_name, ...
    'matrix',              'matrices/WRAN_irreg_384_256.csv', ... % byte-identical to the .mat; repo-side path
    'matrix_matlab',       cfg.matfile, ...
    'z',                   cfg.z, ...
    'state_encoding',      'none', ...           % deterministic baseline: no learned state at all
    'scheduling',          'round_robin_fixed_order', ...
    'eval_maxIter',        cfg.i_max, ...
    'eval_base_seed',      cfg.base_seed, ...
    'target_frame_errors', cfg.target_frame_errors, ...
    'max_frames',          cfg.n_frames, ...
    'seed',                cfg.base_seed, ...
    'n_shards',            nShards, ...
    'chunk_size',          cfg.chunk_size, ...
    'interpreter',         version, ...
    'source',              'external_ref/ldpc_rl/parallel_eval/run_round_robin_eval.m');
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
        c.NumWorkers = cfg.n_workers;
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
    st = round_robin_eval_shard(sh, env);

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

fprintf('\n===== SWEEP COMPLETE (z=%d, %.1fs wall, %.4g s/frame effective) =====\n', ...
    cfg.z, wall, wall/(numel(cfg.snr_db)*cfg.n_frames));
fprintf('%8s %10s %14s %12s %12s\n', 'SNR_dB', 'frames', 'BER', 'FER', 'converged');
for si = 1:numel(summary)
    fprintf('%8g %10d %14.6g %12.6g %12d\n', summary(si).snr_db, summary(si).frames, ...
        summary(si).ber, summary(si).fer, summary(si).converged);
end
end
