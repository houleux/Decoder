function out = fast_ppo_rollout(Lch, env, net, posf, i_max, train, U)
% FAST_PPO_ROLLOUT - run B frames of the PPO cluster scheduler side by side.
%
% Batched version of sched_frame.m's 'ppo' kind: the same decoder, features,
% network and action rule, but all frames still running advance one scheduling
% step together, so features, the scorer and the CN updates are a few large
% array operations instead of B small ones. lib/fast_ppo_smoketest.m checks
% it against sched_frame.m frame by frame.
%
% INPUTS
%   Lch    B x n channel LLRs
%   env    sched_env(code, z)
%   net    PPO weights (ppo_init.m), posf  pos_feature flag
%   train  true: sample actions (reward 'success'), record transitions, a frame
%          ends on the step its syndrome reaches zero.  false: greedy, syndrome
%          checked once per iteration (as sched_frame.m).
%   U      B x (K*i_max) uniforms; frame b's t-th sampled action uses U(b, t)
%          (sched_frame.m draws one rand per training step)
% OUTPUT out.L (B x n), .converged, .iterations, .messages (B x 1)
%        train: out.ep (transitions of frame 1, then frame 2, ...; same fields as
%        sched_frame.m's out.ep), out.T (B x 1 steps per frame), out.reward

[B, n] = size(Lch);
K = env.K;  m = env.m;  H = env.H;
NB = env.NB;  nbLen = env.nbLen(:)';  padmask = env.padmask;  maxNb = env.maxNb;

% CN edge tables, padded: VN n+2 holds +Inf, so a padded edge has tanh(Inf/2) = 1
deg = cellfun(@numel, env.CN_neighbors);  dmax = max(deg);
CNI = (n + 2) * ones(m, dmax);
for c = 1:m, CNI(c, 1:deg(c)) = env.CN_neighbors{c}; end
zmax = max(cellfun(@numel, env.clusters));
CLM = zeros(K, zmax);
for k = 1:K, CLM(k, 1:numel(env.clusters{k})) = env.clusters{k}; end

L = [Lch, zeros(B, 1), inf(B, 1)];       % column n+1 = 0 backs NB's padding
R = zeros(B, m * dmax);                  % CN->VN messages, slot (c-1)*dmax + j
Ht = H';
D = size(net.W1, 2);
messages = zeros(B, 1);  converged = false(B, 1);  iters = i_max * ones(B, 1);
act = true(B, 1);
if train
    T = zeros(B, 1);  Tm = K * i_max;
    eX = zeros(K, D, Tm, B, 'single');  eM = false(K, Tm, B);
    eA = zeros(Tm, B);  eLP = zeros(Tm, B);  eV = zeros(Tm, B);  eR = zeros(Tm, B);  eD = false(Tm, B);
end

for it = 1:i_max
    avail = true(B, K);
    for step = 1:K
        bs = find(act);  nb_ = numel(bs);
        if nb_ == 0, break; end
        Lb = L(bs, :);
        %% features (ppo_features.m, one row block per frame)
        unsat = mod(double(Lb(:, 1:n) < 0) * Ht, 2);                 % nb_ x m
        Lx = reshape(Lb(:, NB(:)'), nb_, K, maxNb);
        f0 = sum(tanh(min(max(Lx, -30), 30)), 3) ./ nbLen;
        f1 = unsat * env.Cavg';
        A2 = abs(tanh(Lx / 2));
        A2(repmat(reshape(padmask, 1, K, maxNb), nb_, 1, 1)) = Inf;
        f2 = min(A2, [], 3);
        z0 = (f0 - mean(f0, 2)) ./ (std(f0, 1, 2) + 1e-6);
        z2 = (f2 - mean(f2, 2)) ./ (std(f2, 1, 2) + 1e-6);
        av = avail(bs, :);
        g = [(it - 1) / i_max * ones(nb_, 1), (step - 1) / K * ones(nb_, 1), mean(unsat, 2)];
        cols = {f0, f1, f2, z0, z2, double(av)};
        if posf, cols{end + 1} = repmat(env.pos(:)', nb_, 1); end
        for j = 1:3, cols{end + 1} = repmat(g(:, j), 1, K); end   %#ok<AGROW>
        Xf = zeros(K * nb_, D);                                  % row k + (b-1)*K
        for j = 1:D, Xf(:, j) = reshape(cols{j}', [], 1); end
        %% scorer
        h = tanh(Xf * net.W1' + net.b1);
        h = tanh(h * net.W2' + net.b2);
        lg = reshape(h * net.W3' + net.b3, K, nb_);
        lg(~av') = -Inf;
        if train
            zs = lg - max(lg, [], 1);
            p = exp(zs);  sp = sum(p, 1);  p = p ./ sp;
            u = U(sub2ind(size(U), bs', T(bs)' + 1));
            a = sum(cumsum(p, 1) < u, 1) + 1;
            for i = find(a > K)                                     % rounding: as sched_frame.m
                a(i) = find(av(i, :), 1, 'last');
            end
            lin = a + (0:nb_-1) * K;
            logp = zs(lin) - log(sp);
            e = tanh(tanh(Xf * net.E1' + net.c1) * net.E2' + net.c2);
            ebar = reshape(mean(reshape(e, K, nb_, []), 1), nb_, []);
            v = tanh([ebar, g] * net.C1' + net.d1) * net.C2' + net.d2;
        else
            [~, a] = max(lg, [], 1);
        end
        avail(sub2ind([B, K], bs', a)) = false;
        %% schedule: CN -> VN for each check of the chosen cluster, in index order
        for j = 1:zmax
            cn = CLM(a, j);
            ok = cn > 0;
            if ~any(ok), continue; end
            rb = bs(ok);  cn = cn(ok);
            I = CNI(cn, :);
            li = rb + (I - 1) * B;
            ri = rb + ((cn - 1) * dmax + (0:dmax-1)) * B;
            qv = L(li) - R(ri);
            t = tanh(qv / 2);
            sm = abs(t) < 1e-12;
            if any(sm(:))
                t(sm & t >= 0) = 1e-12;
                t(sm & t < 0)  = -1e-12;
            end
            pr = prod(t, 2) ./ t;
            pr = min(max(pr, -(1 - 1e-15)), 1 - 1e-15);
            r = log((1 + pr) ./ (1 - pr));
            R(ri) = r;
            L(li) = qv + r;
            L(:, n + 2) = Inf;                     % padded edges wrote Inf + r
        end
        messages(bs) = messages(bs) + env.clMsgs(a);
        %% transitions / termination
        if train
            conv = ~any(mod(double(L(bs, 1:n) < 0) * Ht, 2), 2);
            for i = 1:nb_
                b = bs(i);  T(b) = T(b) + 1;  t_ = T(b);
                eX(:, :, t_, b) = reshape(Xf((i-1)*K + (1:K), :), K, D);
                eM(:, t_, b) = av(i, :)';  eA(t_, b) = a(i);  eLP(t_, b) = logp(i);
                eV(t_, b) = v(i);  eR(t_, b) = double(conv(i));
                eD(t_, b) = conv(i) || (it == i_max && step == K);
            end
            done = bs(conv);
            converged(done) = true;  iters(done) = it;  act(done) = false;
        end
    end
    if ~train
        bs = find(act);
        conv = ~any(mod(double(L(bs, 1:n) < 0) * Ht, 2), 2);
        converged(bs(conv)) = true;  iters(bs(conv)) = it;  act(bs(conv)) = false;
    end
end

out = struct('L', L(:, 1:n), 'converged', converged, 'iterations', iters, 'messages', messages);
if train
    Ttot = sum(T);
    X = zeros(K, D, Ttot, 'single');  M = false(K, Ttot);
    [A, LP, V, RR, DD] = deal(zeros(1, Ttot), zeros(1, Ttot), zeros(1, Ttot), zeros(1, Ttot), false(1, Ttot));
    o = 0;
    for b = 1:B
        s = o + (1:T(b));
        X(:, :, s) = eX(:, :, 1:T(b), b);  M(:, s) = eM(:, 1:T(b), b);
        A(s) = eA(1:T(b), b);  LP(s) = eLP(1:T(b), b);  V(s) = eV(1:T(b), b);
        RR(s) = eR(1:T(b), b);  DD(s) = eD(1:T(b), b);
        o = o + T(b);
    end
    out.ep = struct('X', X, 'mask', M, 'a', A, 'logp', LP, 'v', V, 'r', RR, 'done', DD);
    out.T = T;  out.reward = double(converged);
end
end
