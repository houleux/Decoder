function [out, pol] = sched_frame(Lch, env, pol, opt)
% SCHED_FRAME - decode (or train on) ONE frame with cluster-sequential BP.
%
% Scheduling rule, identical for every policy and identical to this repo's
% Python rl/trainer.py / rl/agents/reldec.py / global_mdp PPO agent: each of
% the i_max iterations schedules every cluster exactly once; the policy only
% chooses the ORDER. Scheduling a cluster runs the product-sum CN->VN update
% for each of its check nodes in increasing index order.
%
% pol.kind
%   'round_robin'  clusters 1..K in order
%   'reldec'       tabular Q, state = hard decisions (L > 0) of the cluster's
%                  neighborhood, reward = fraction of them that are positive
%   'tanh_mi'      tabular Q, state = mean tanh(L) over the neighborhood in 21
%                  bins, reward = increase in the neighborhood's mean MI
%   'ppo'          shared per-cluster scorer network (see ppo_features.m)
% Tabular policies pick the max-Q available cluster, ties broken uniformly at
% random; unseen states have Q = 0.
%
% opt.i_max   iterations
% opt.train   false: greedy, syndrome checked once per iteration.
%             true (tabular): epsilon-greedy + Q-learning update after every
%               cluster, bootstrapped on that cluster's own next state.
%             true (ppo): sample from the policy, record the episode in
%               out.ep, syndrome checked after every step (episode ends there).
%
% CN update numerics match the Python decoders: |tanh| floored at 1e-12, the
% leave-one-out product capped at 1-1e-15, message = log((1+p)/(1-p)).
%
% OUTPUT out.L (1 x n), .converged, .iterations, .messages, .reward, .ep

n = env.n;  K = env.K;  H = env.H;
CN = env.CN_neighbors;  clusters = env.clusters;
NB = env.NB;  nbLen = env.nbLen;  nb = env.nb;
L = [Lch(:)', 0];          % L(n+1) = 0 backs the padded entries of NB
res = env.res0;
i_max = opt.i_max;  train = opt.train;

switch pol.kind
    case 'round_robin', kind = 1;
    case 'reldec',      kind = 2;
    case 'tanh_mi',     kind = 3;
    case 'ppo',         kind = 4;
    otherwise, error('sched_frame:badKind', 'unknown policy kind "%s"', pol.kind);
end
tab = (kind == 2 || kind == 3);
if tab
    dense = pol.dense;
    if dense, nS = size(pol.Q, 1);  colOff = (0:K-1)' * nS; end
    if kind == 2, pow2 = pol.pow2; end
    % 'map' store (RELDEC, neighborhoods > 52 bits): one containers.Map per
    % cluster, keyed by the exact bit pattern as a '0'/'1' string; only visited
    % states are stored, unseen states read as Q = 0.
    usemap = kind == 2 && isfield(pol, 'store') && strcmp(pol.store, 'map');
end
if kind == 4
    net = pol.net;  posf = pol.pos_feature;
    if train
        cap = 2048;  D = size(net.W1, 2);
        epX = zeros(K, D, cap, 'single');  epM = false(K, cap);
        epA = zeros(1, cap);  epLP = zeros(1, cap);  epV = zeros(1, cap);
        epR = zeros(1, cap);  epD = false(1, cap);  T = 0;
    end
end

messages = 0;  total_reward = 0;  converged = false;  it_done = i_max;

for it = 1:i_max
    avail = true(1, K);
    for step = 1:K
        %% ---------------- choose a cluster ----------------
        if kind == 1
            a = step;
        elseif tab
            if kind == 2 && usemap
                B = L(NB) > 0;
            elseif kind == 2
                key = double(L(NB) > 0) * pow2;
            else
                x = sum(tanh(L(NB)), 2) ./ nbLen;
                key = floor((min(max(x, -1), 1) + 1) / 2 * 20 + 0.5);
            end
            if train && rand < pol.epsilon
                av = find(avail);
                a = av(randi(numel(av)));
            else
                if dense
                    q = pol.Q(key + 1 + colOff);
                elseif usemap
                    q = zeros(K, 1);
                    for j = find(avail)
                        ks = char('0' + B(j, 1:nbLen(j)));
                        if isKey(pol.maps{j}, ks), q(j) = pol.maps{j}(ks); end
                    end
                else
                    q = zeros(K, 1);
                    for j = find(avail)
                        c = pol.cnt(j);
                        if c > 0
                            i = find(pol.keys(1:c, j) == key(j), 1);
                            if ~isempty(i), q(j) = pol.vals(i, j); end
                        end
                    end
                end
                q(~avail) = -Inf;
                ties = find(q == max(q));
                if numel(ties) == 1
                    a = ties;
                else
                    a = ties(randi(numel(ties)));
                end
            end
            if usemap
                sb = char('0' + B(a, 1:nbLen(a)));
            else
                sb = key(a);
            end
            if train && kind == 3, Lb = L(nb{a}); end
        else
            X = ppo_features(L, env, avail, it, i_max, step, posf);
            h = tanh(X * net.W1' + net.b1);
            h = tanh(h * net.W2' + net.b2);
            lg = h * net.W3' + net.b3;
            lg(~avail) = -Inf;
            if train
                zs = lg - max(lg);
                p = exp(zs);  sp = sum(p);  p = p / sp;
                a = find(cumsum(p) >= rand, 1);
                if isempty(a), a = find(avail, 1, 'last'); end
                logp = zs(a) - log(sp);
                e = tanh(tanh(X * net.E1' + net.c1) * net.E2' + net.c2);
                v = tanh([mean(e, 1), X(1, end-2:end)] * net.C1' + net.d1) * net.C2' + net.d2;
                mask = avail;
                if strcmp(pol.reward, 'errors_success'), eb = nnz(L(nb{a}) < 0); end
            else
                [~, a] = max(lg);
            end
        end
        avail(a) = false;

        %% ---------------- schedule it: CN -> VN for each of its checks ----------------
        for cn = clusters{a}
            idx = CN{cn};
            qv = L(idx) - res{cn};
            t = tanh(qv / 2);
            sm = abs(t) < 1e-12;
            if any(sm)
                t(sm & t >= 0) = 1e-12;
                t(sm & t < 0)  = -1e-12;
            end
            pr = prod(t) ./ t;
            pr = min(max(pr, -(1 - 1e-15)), 1 - 1e-15);
            r = log((1 + pr) ./ (1 - pr));
            res{cn} = r;
            L(idx) = qv + r;
        end
        messages = messages + env.clMsgs(a);

        %% ---------------- learning ----------------
        if train && tab
            La = L(nb{a});
            if kind == 2
                pos = La > 0;
                reward = nnz(pos) / nbLen(a);
                if usemap
                    sa = char('0' + pos);
                else
                    sa = double(pos) * pow2(1:nbLen(a));
                end
            else
                reward = mean(mi_llr(La)) - mean(mi_llr(Lb));
                x = mean(tanh(La));
                sa = floor((min(max(x, -1), 1) + 1) / 2 * 20 + 0.5);
            end
            total_reward = total_reward + reward;
            if dense
                i1 = sb + 1 + (a - 1) * nS;  i2 = sa + 1 + (a - 1) * nS;
                pol.Q(i1) = pol.Q(i1) + pol.alpha * (reward + pol.gamma * pol.Q(i2) - pol.Q(i1));
            elseif usemap
                M = pol.maps{a};                   % handle: updated in place
                q1 = 0;  q2 = 0;
                if isKey(M, sb), q1 = M(sb); end
                if isKey(M, sa), q2 = M(sa); end
                M(sb) = q1 + pol.alpha * (reward + pol.gamma * q2 - q1);
            else
                c = pol.cnt(a);
                i1 = find(pol.keys(1:c, a) == sb, 1);
                i2 = find(pol.keys(1:c, a) == sa, 1);
                if isempty(i2), q2 = 0; else, q2 = pol.vals(i2, a); end
                if isempty(i1)
                    c = c + 1;
                    if c > size(pol.keys, 1)
                        pol.keys = [pol.keys; zeros(size(pol.keys))];
                        pol.vals = [pol.vals; zeros(size(pol.vals))];
                    end
                    pol.cnt(a) = c;  pol.keys(c, a) = sb;  i1 = c;
                end
                q1 = pol.vals(i1, a);
                pol.vals(i1, a) = q1 + pol.alpha * (reward + pol.gamma * q2 - q1);
            end
        elseif train && kind == 4
            conv = all(mod(H * double(L(1:n) < 0)', 2) == 0);
            if strcmp(pol.reward, 'success')
                reward = double(conv);
            else
                reward = (eb - nnz(L(nb{a}) < 0)) / n + double(conv);
            end
            total_reward = total_reward + reward;
            T = T + 1;
            if T > cap
                epX = cat(3, epX, zeros(K, D, cap, 'single'));  epM = [epM, false(K, cap)];
                epA = [epA, zeros(1, cap)];  epLP = [epLP, zeros(1, cap)];  epV = [epV, zeros(1, cap)];
                epR = [epR, zeros(1, cap)];  epD = [epD, false(1, cap)];  cap = 2 * cap;
            end
            epX(:, :, T) = X;  epM(:, T) = mask(:);  epA(T) = a;  epLP(T) = logp;
            epV(T) = v;  epR(T) = reward;
            epD(T) = conv || (it == i_max && step == K);
            if conv
                converged = true;  it_done = it;
                break;
            end
        end
    end
    if converged, break; end
    if ~(train && kind == 4)
        if all(mod(H * double(L(1:n) < 0)', 2) == 0)
            converged = true;  it_done = it;
            break;
        end
    end
end

out = struct('L', L(1:n), 'converged', converged, 'iterations', it_done, ...
             'messages', messages, 'reward', total_reward);
if kind == 4 && train
    out.ep = struct('X', epX(:, :, 1:T), 'mask', epM(:, 1:T), 'a', epA(1:T), ...
                    'logp', epLP(1:T), 'v', epV(1:T), 'r', epR(1:T), 'done', epD(1:T));
end
end
