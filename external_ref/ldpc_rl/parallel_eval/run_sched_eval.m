function summary = run_sched_eval(cfg_json)
% RUN_SCHED_EVAL - parallel, resumable BER/FER sweep for any cluster-scheduling
% decoder (flooding, round_robin, reldec, tanh_mi, ppo), written to the spool
% that matlab_bridge.ingest commits to the experiment database.
%
% Generalises run_round_robin_eval.m (same shard / substream / spool scheme).
% The whole configuration comes from one JSON file written by the Python
% driver (matlab_bridge/sched_driver.py), so the manifest the ingester hashes
% into a config_id is exactly the one the driver checked against the database.
%
% cfg_json fields
%   manifest   object copied verbatim to <spool_dir>/manifest.json (must carry
%              method with the matlab_ prefix, matrix, z, seed, state_encoding,
%              scheduling, n_shards is filled in here)
%   code, z, kind ('flooding'|'round_robin'|'reldec'|'tanh_mi'|'ppo')
%   checkpoint (learned kinds) .mat written by train/sched_train.m
%   snr_db, n_frames, chunk_size, i_max, base_seed, n_workers, spool_dir
%
% Optional cfg.part / cfg.nparts: process only shards j with
% mod(j-1, nparts) == part-1, so N independent processes (e.g. Octave, whose
% parfor is serial) can share one sweep. In that mode the manifest is left to
% the driver, which wrote it before starting the processes.
%
% Resume: a shard whose chunk file already exists in spool_dir or
% spool_dir/done is not recomputed. Substreams depend only on the shard list,
% so a restarted sweep reproduces exactly the frames it skipped.

here = fileparts(mfilename('fullpath'));
addpath(fullfile(here, '..', 'lib'));
cfg = jsondecode(fileread(cfg_json));
if ~startsWith(cfg.manifest.method, 'matlab_')
    error('run_sched_eval:badMethodName', 'manifest.method must start with matlab_ (got %s)', cfg.manifest.method);
end

env = sched_env(cfg.code, cfg.z);
switch cfg.kind
    case {'flooding', 'round_robin'}
        pol = struct('kind', cfg.kind);
    case {'reldec', 'tanh_mi', 'ppo'}
        ck = load(cfg.checkpoint, 'pol', 'meta');
        pol = sched_policy_unpack(ck.pol);
        if ~strcmp(pol.kind, cfg.kind) || ck.meta.z ~= cfg.z || ~strcmp(ck.meta.code, cfg.code)
            error('run_sched_eval:ckptMismatch', 'checkpoint %s is %s/%s/z=%d, expected %s/%s/z=%d', ...
                cfg.checkpoint, ck.meta.code, pol.kind, ck.meta.z, cfg.code, cfg.kind, cfg.z);
        end
    otherwise
        error('run_sched_eval:badKind', 'unknown kind %s', cfg.kind);
end

snrs = cfg.snr_db(:)';
shards = struct('substream', {}, 'snr_db', {}, 'n_frames', {}, 'chunk_id', {});
sub = 0;
for si = 1:numel(snrs)
    remaining = cfg.n_frames;  ci = 0;
    while remaining > 0
        nf = min(cfg.chunk_size, remaining);
        sub = sub + 1;
        shards(end+1) = struct('substream', sub, 'snr_db', snrs(si), 'n_frames', nf, ...
            'chunk_id', sprintf('%s_snr%g_c%06d', cfg.kind, snrs(si), ci)); %#ok<AGROW>
        remaining = remaining - nf;  ci = ci + 1;
    end
end
nShards = numel(shards);

spool = cfg.spool_dir;
if ~exist(spool, 'dir'), mkdir(spool); end
manifest = cfg.manifest;
manifest.n_shards = nShards;
manifest.chunk_size = cfg.chunk_size;
manifest.eval_base_seed = cfg.base_seed;
manifest.interpreter = version;
manifest.source = 'external_ref/ldpc_rl/parallel_eval/run_sched_eval.m';
split = isfield(cfg, 'nparts') && cfg.nparts > 1;
if ~split
    fid = fopen(fullfile(spool, 'manifest.json'), 'w');
    fprintf(fid, '%s', jsonencode(manifest));
    fclose(fid);
end

todo = true(1, nShards);
for i = 1:nShards
    f = [shards(i).chunk_id '.json'];
    todo(i) = ~exist(fullfile(spool, f), 'file') && ~exist(fullfile(spool, 'done', f), 'file');
end
if split
    todo = todo & (mod((1:nShards) - 1, cfg.nparts) == cfg.part - 1);
end
idx = find(todo);
fprintf('run_sched_eval: %s %s z=%d (%d clusters) | SNR %s | %d frames/SNR | %d/%d shards to run\n', ...
    manifest.method, cfg.code, cfg.z, env.K, mat2str(snrs), cfg.n_frames, numel(idx), nShards);

isOct = exist('OCTAVE_VERSION', 'builtin') ~= 0;
if ~isOct && ~split
    % Private, node-local job storage: two MATLAB pools (training and evaluation
    % jobs) running at once must not share ~/.matlab's default local-cluster storage.
    c = parcluster('local');
    jsl = tempname;  mkdir(jsl);
    c.JobStorageLocation = jsl;
    c.NumWorkers = cfg.n_workers;
    parpool(c, cfg.n_workers);
end

sh = shards(idx);  i_max = cfg.i_max;  bseed = cfg.base_seed;
tfe = manifest.target_frame_errors;  maxf = manifest.max_frames;  meth = manifest.method;
t0 = tic;
parfor j = 1:numel(sh)
    st = sched_eval_shard(sh(j), env, pol, i_max, bseed);
    payload = struct('method', meth, 'snr_db', sh(j).snr_db, 'target_frame_errors', tfe, ...
        'max_frames', maxf, 'frames', st.frames, 'bit_errors', st.bit_errors, ...
        'total_bits', st.total_bits, 'frame_errors', st.frame_errors, 'messages', st.messages, ...
        'converged_frames', st.converged_frames, 'iterations', st.iterations, ...
        'substream', sh(j).substream, 'wall_s', st.wall_s);
    spool_write_chunk(spool, sh(j).chunk_id, payload);
end
fprintf('run_sched_eval: %d shards in %.1fs\n', numel(sh), toc(t0));
summary = numel(sh);
end
