function sched_train_batch(list_json, n_workers)
% SCHED_TRAIN_BATCH - run several train/sched_train.m configs in parallel, one
% per parpool worker (each training is single-threaded). list_json holds a
% JSON array of sched_train cfg objects; finished checkpoints are skipped.
here = fileparts(mfilename('fullpath'));
addpath(fullfile(here, '..', 'lib'));
cfgs = jsondecode(fileread(list_json));
if isstruct(cfgs), cfgs = num2cell(cfgs); end
isOct = exist('OCTAVE_VERSION', 'builtin') ~= 0;
if ~isOct
    % Private, node-local job storage: two MATLAB pools (training and evaluation
    % jobs) running at once must not share ~/.matlab's default local-cluster storage.
    c = parcluster('local');
    jsl = tempname;  mkdir(jsl);
    c.JobStorageLocation = jsl;
    c.NumWorkers = n_workers;
    parpool(c, n_workers);
end
fprintf('sched_train_batch: %d configs on %d workers\n', numel(cfgs), n_workers);
parfor i = 1:numel(cfgs)
    c = cfgs{i};
    c.train_snrs = c.train_snrs(:)';
    try
        sched_train(c);
    catch err
        fprintf(2, 'sched_train_batch: %s FAILED: %s\n', c.out, getReport(err));
    end
end
end
