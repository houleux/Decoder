function X = ppo_features(L, env, avail, it, i_max, step, pos_feature)
% PPO_FEATURES - observation for the PPO cluster scorer: one row per cluster.
%
% Same definition as this repo's Python
% global_mdp/agents/global_ppo_agent.py::GlobalPPOAgent._obs.
% Per cluster k:
%   1  mean over its neighborhood of tanh(clip(L, -30, 30))
%   2  fraction of its check nodes that are unsatisfied (hard decisions L < 0)
%   3  min over its neighborhood of |tanh(L / 2)|
%   4  column 1 z-scored across clusters   (population std, + 1e-6)
%   5  column 3 z-scored across clusters
%   6  1 if the cluster is still available this iteration, else 0
%   7  k/(K-1), only when pos_feature is true
% followed by three global columns repeated on every row:
%   (it-1)/i_max,  (step-1)/K,  fraction of all checks unsatisfied
%
% INPUTS  L (1 x n+1, last entry 0), env (sched_env), avail (1 x K logical),
%         it (1-based iteration), i_max, step (1-based step within iteration)

n = env.n;  K = env.K;
unsat = mod(env.H * double(L(1:n) < 0)', 2);
Lx = L(env.NB);
f0 = sum(tanh(min(max(Lx, -30), 30)), 2) ./ env.nbLen;
f1 = env.Cavg * unsat;
A2 = abs(tanh(Lx / 2));
A2(env.padmask) = Inf;
f2 = min(A2, [], 2);
z0 = (f0 - mean(f0)) / (std(f0, 1) + 1e-6);
z2 = (f2 - mean(f2)) / (std(f2, 1) + 1e-6);
g  = [(it - 1) / i_max, (step - 1) / K, mean(unsat)];
if pos_feature
    X = [f0, f1, f2, z0, z2, double(avail(:)), env.pos, repmat(g, K, 1)];
else
    X = [f0, f1, f2, z0, z2, double(avail(:)), repmat(g, K, 1)];
end
end
