function [loss, g, info] = ppo_lossgrad(net, Xb, maskb, a, oldlogp, adv, ret, hp)
% PPO_LOSSGRAD - PPO minibatch loss and its gradient, backpropagated by hand.
%
% Same network and loss as this repo's Python global_mdp/algorithms/ppo.py:
%   actor   one MLP shared by all clusters, [features_k, global] -> 32 -> 32 -> 1
%           (tanh), softmax over the AVAILABLE clusters only
%   critic  shared embedding [features_k, global] -> 32 -> 32 (tanh), mean over
%           clusters, concatenated with the 3 global features -> 32 -> 1
%   loss  = -mean(min(r*A, clip(r, 1-c, 1+c)*A)) + vf*mean((V-R)^2) - ent*mean(H)
% lib/ppo_gradcheck.m verifies the gradient against finite differences.
%
% INPUTS
%   Xb      K x D x B observations        maskb   K x B logical availability
%   a       1 x B chosen clusters         oldlogp 1 x B log-prob at collection
%   adv     1 x B normalised advantages   ret     1 x B return targets
%   hp      .clip_eps .vf_coef .ent_coef
% OUTPUTS  loss, g (struct like net), info (.pg .v .ent)

[K, D, B] = size(Xb);
hdim = size(net.W1, 1);
Xf = reshape(permute(double(Xb), [1 3 2]), K*B, D);     % row index = k + (b-1)*K

%% actor forward
A1 = tanh(Xf * net.W1' + net.b1);
A2 = tanh(A1 * net.W2' + net.b2);
S  = reshape(A2 * net.W3' + net.b3, K, B);
S(~maskb) = -Inf;
Zs = S - max(S, [], 1);
P  = exp(Zs);  sP = sum(P, 1);  P = P ./ sP;
logP = Zs - log(sP);
idx  = a + (0:B-1) * K;
logp = logP(idx);
ratio = exp(logp - oldlogp);
s1 = ratio .* adv;
s2 = min(max(ratio, 1 - hp.clip_eps), 1 + hp.clip_eps) .* adv;
pg = -mean(min(s1, s2));
PlogP = P .* logP;  PlogP(~maskb) = 0;
Hent = -sum(PlogP, 1);
ent  = mean(Hent);

%% critic forward
E1a = tanh(Xf * net.E1' + net.c1);
E2a = tanh(E1a * net.E2' + net.c2);
ebar = reshape(mean(reshape(E2a, K, B, hdim), 1), B, hdim);
G    = reshape(Xb(1, D-2:D, :), 3, B)';
Cin  = [ebar, double(G)];
Hc   = tanh(Cin * net.C1' + net.d1);
V    = (Hc * net.C2' + net.d2)';
vl   = mean((V - ret).^2);

loss = pg + hp.vf_coef * vl - hp.ent_coef * ent;
info = struct('pg', pg, 'v', vl, 'ent', ent);

%% actor backward
% d(min(s1,s2))/d(ratio) = adv where the unclipped term is active: inside the
% clip range (both terms equal) or where it is the smaller one.
active = (ratio >= 1 - hp.clip_eps & ratio <= 1 + hp.clip_eps) | (s1 < s2);
dlogp = -(adv .* ratio .* active) / B;
dS = -P .* dlogp;                          % d logp_a / dS_k = [k == a] - P_k
dS(idx) = dS(idx) + dlogp;
Tm = P .* (logP + Hent);  Tm(~maskb) = 0;  % dH/dS_k = -P_k (logP_k + H)
dS = dS + (hp.ent_coef / B) * Tm;
dS(~maskb) = 0;
dOut = reshape(dS, K*B, 1);
g.W3 = dOut' * A2;               g.b3 = sum(dOut);
dZ2  = (dOut * net.W3) .* (1 - A2.^2);
g.W2 = dZ2' * A1;                g.b2 = sum(dZ2, 1);
dZ1  = (dZ2 * net.W2) .* (1 - A1.^2);
g.W1 = dZ1' * Xf;                g.b1 = sum(dZ1, 1);

%% critic backward
dV   = hp.vf_coef * 2 * (V - ret) / B;     % 1 x B
g.C2 = dV * Hc;                  g.d2 = sum(dV);
dZc  = (dV' * net.C2) .* (1 - Hc.^2);
g.C1 = dZc' * Cin;               g.d1 = sum(dZc, 1);
dCin = dZc * net.C1;
debar = dCin(:, 1:hdim) / K;                                   % B x h
dE2a = reshape(repmat(reshape(debar, 1, B, hdim), K, 1, 1), K*B, hdim);
dZe2 = dE2a .* (1 - E2a.^2);
g.E2 = dZe2' * E1a;              g.c2 = sum(dZe2, 1);
dZe1 = (dZe2 * net.E2) .* (1 - E1a.^2);
g.E1 = dZe1' * Xf;               g.c1 = sum(dZe1, 1);
end
