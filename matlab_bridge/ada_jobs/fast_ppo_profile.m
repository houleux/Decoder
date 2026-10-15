% FAST_PPO_PROFILE - MATLAB smoke test of fast_ppo_rollout.m, then training-speed
% profile of ppo vs fast_ppo on WRAN (run from external_ref/ldpc_rl).
cd(fullfile(getenv('HOME'), 'Decoder_100k', 'external_ref', 'ldpc_rl'));   % run() starts in this file's folder
addpath('lib'); addpath('train');
fast_ppo_smoketest();
base = struct('lr', 1e-3, 'gamma', 1, 'gae_lambda', .95, 'clip_eps', .2, 'update_epochs', 4, ...
    'minibatch_size', 512, 'rollout_steps', 8192, 'ent_coef', .003, 'vf_coef', .5, 'max_grad_norm', .5, ...
    'hidden', 32, 'reward', 'success', 'pos_feature', false);
out = tempname;  mkdir(out);
runs = { % z, method, episodes, update_epochs, train_snrs, collect_workers, threads
    4,  'ppo',      1000, 4, [1 1.5 2 2.5 3], 1, 1
    4,  'fast_ppo', 2000, 4, [1 1.5 2 2.5 3], 1, 1
    4,  'fast_ppo', 2000, 2, [1.5 2 2.5 3],   1, 1
    4,  'fast_ppo', 2000, 2, [1.5 2 2.5 3],   1, 4
    4,  'fast_ppo', 2000, 2, [1.5 2 2.5 3],   4, 1
    4,  'fast_ppo', 2000, 2, [1.5 2 2.5 3],   8, 1
    16, 'ppo',      2000, 4, [1 1.5 2 2.5 3], 1, 1
    16, 'fast_ppo', 4000, 2, [1.5 2 2.5 3],   1, 1
    16, 'fast_ppo', 4000, 2, [1.5 2 2.5 3],   4, 1};
res = {};
for i = 1:size(runs, 1)
    [z, meth, eps, ue, snrs, W, thr] = runs{i, :};
    hp = base;  hp.update_epochs = ue;
    if strcmp(meth, 'fast_ppo'), hp.batch_frames = 64;  hp.collect_workers = W; end
    delete(gcp('nocreate'));
    c = struct('code', 'wran', 'z', z, 'method', meth, 'episodes', eps, 'l_max', 5, 'train_snrs', snrs, ...
               'seed', 42, 'out', fullfile(out, sprintf('r%d.mat', i)), 'ppo', hp, 'threads', thr);
    if W > 1, delete(gcp('nocreate')); end
    t = tic;  ck = sched_train(c);  wall = toc(t);
    cs = NaN;  us = NaN;
    if isfield(ck.meta, 'collect_s'), cs = ck.meta.collect_s;  us = ck.meta.update_s; end
    res(end + 1, :) = {z, meth, ue, numel(snrs), W, thr, ck.meta.episodes, wall, 1000 * wall / ck.meta.episodes, cs, us}; %#ok<SAGROW>
end
delete(gcp('nocreate'));
fprintf('\nPROFILE  z  method    epochs nSNR  W thr  episodes  wall_s  ms/episode  collect_s  update_s\n');
for i = 1:size(res, 1)
    fprintf('PROFILE %2d  %-9s %d      %d    %d  %d   %6d  %7.1f  %8.2f  %8.1f  %8.1f\n', res{i, :});
end
