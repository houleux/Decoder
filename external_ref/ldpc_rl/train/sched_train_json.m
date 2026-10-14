function sched_train_json(cfg_json)
% SCHED_TRAIN_JSON - run train/sched_train.m on one cfg stored as JSON (used to
% launch each training as its own process, e.g. under Octave).
here = fileparts(mfilename('fullpath'));
addpath(here);  addpath(fullfile(here, '..', 'lib'));
c = jsondecode(fileread(cfg_json));
c.train_snrs = c.train_snrs(:)';
sched_train(c);
end
