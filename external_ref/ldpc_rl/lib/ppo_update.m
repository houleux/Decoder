function [net, opt, info] = ppo_update(net, opt, buf, N, hp)
% PPO_UPDATE - one PPO update over the first N transitions of a rollout buffer.
% GAE advantages (normalised over the rollout), then hp.update_epochs passes of
% shuffled minibatches; each step clips the global gradient norm and applies
% Adam (PyTorch defaults: beta 0.9/0.999, eps 1e-8). Mirrors
% global_mdp/algorithms/ppo.py::PPO.update. The rollout must end on a terminal
% step, so no value bootstrap is needed.
%
% buf fields (first N entries used): X (K x D x cap single), mask (K x cap),
%   a, logp, v, r, done (1 x cap)
% opt: Adam state (.m .v structs like net, .t); pass [] to start fresh.

if ~buf.done(N)
    error('ppo_update:midEpisode', 'rollout must end on a terminal step');
end
r = buf.r(1:N);  v = buf.v(1:N);  dn = double(buf.done(1:N));
adv = zeros(1, N);  gae = 0;
for t = N:-1:1
    if dn(t), nextv = 0; else, nextv = v(t + 1); end
    delta = r(t) + hp.gamma * nextv - v(t);
    gae = delta + hp.gamma * hp.gae_lambda * (1 - dn(t)) * gae;
    adv(t) = gae;
end
ret = adv + v;
adv = (adv - mean(adv)) / (std(adv) + 1e-8);

names = fieldnames(net);
if isempty(opt)
    opt.t = 0;
    for i = 1:numel(names)
        opt.m.(names{i}) = zeros(size(net.(names{i})));
        opt.v.(names{i}) = zeros(size(net.(names{i})));
    end
end
b1 = 0.9;  b2 = 0.999;  epsA = 1e-8;
sumL = 0;  nmb = 0;
for ep = 1:hp.update_epochs
    perm = randperm(N);
    for s = 1:hp.minibatch_size:N
        id = perm(s:min(s + hp.minibatch_size - 1, N));
        [loss, g] = ppo_lossgrad(net, buf.X(:, :, id), buf.mask(:, id), buf.a(id), ...
                                 buf.logp(id), adv(id), ret(id), hp);
        gn2 = 0;
        for i = 1:numel(names), gn2 = gn2 + sum(g.(names{i})(:).^2); end
        gn = sqrt(gn2);
        sc = 1;
        if gn > hp.max_grad_norm, sc = hp.max_grad_norm / (gn + 1e-6); end
        opt.t = opt.t + 1;
        for i = 1:numel(names)
            f = names{i};
            gi = g.(f) * sc;
            opt.m.(f) = b1 * opt.m.(f) + (1 - b1) * gi;
            opt.v.(f) = b2 * opt.v.(f) + (1 - b2) * gi.^2;
            mh = opt.m.(f) / (1 - b1^opt.t);
            vh = opt.v.(f) / (1 - b2^opt.t);
            net.(f) = net.(f) - hp.lr * mh ./ (sqrt(vh) + epsA);
        end
        sumL = sumL + loss;  nmb = nmb + 1;
    end
end
info = struct('mean_loss', sumL / nmb, 'minibatches', nmb);
end
