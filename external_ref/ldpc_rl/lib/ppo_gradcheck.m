function maxrel = ppo_gradcheck()
% PPO_GRADCHECK - compare ppo_lossgrad's hand-written gradient with central
% finite differences on a small random problem. Returns the largest relative
% error over the sampled weights (should be ~1e-6 or below).
K = 6;  F = 7;  D = F + 3;  B = 5;  h = 8;
net = ppo_init(D, h);
Xb = randn(K, D, B);
for b = 1:B, Xb(:, D-2:D, b) = repmat(randn(1, 3), K, 1); end   % globals equal across clusters
maskb = rand(K, B) < 0.7;  maskb(1, :) = true;
a = zeros(1, B);
for b = 1:B, av = find(maskb(:, b));  a(b) = av(randi(numel(av))); end
oldlogp = -1.5 + 0.8*randn(1, B);  adv = randn(1, B);  ret = randn(1, B);
hp = struct('clip_eps', 0.2, 'vf_coef', 0.5, 'ent_coef', 0.003);
[~, g] = ppo_lossgrad(net, Xb, maskb, a, oldlogp, adv, ret, hp);
names = fieldnames(net);  maxrel = 0;  d = 1e-6;
for i = 1:numel(names)
    f = names{i};
    for trial = 1:4
        j = randi(numel(net.(f)));
        np = net;  np.(f)(j) = np.(f)(j) + d;
        nm = net;  nm.(f)(j) = nm.(f)(j) - d;
        lp = ppo_lossgrad(np, Xb, maskb, a, oldlogp, adv, ret, hp);
        lm = ppo_lossgrad(nm, Xb, maskb, a, oldlogp, adv, ret, hp);
        num = (lp - lm) / (2*d);
        rel = abs(num - g.(f)(j)) / max(1e-8, abs(num) + abs(g.(f)(j)));
        maxrel = max(maxrel, rel);
    end
end
end
