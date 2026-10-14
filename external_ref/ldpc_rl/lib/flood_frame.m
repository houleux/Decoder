function out = flood_frame(Lch, env, i_max)
% FLOOD_FRAME - standard flooding (parallel) product-sum BP for one frame.
%
% Matches this repo's Python rl/decoder/flooding.py: every check updates from
% the previous iteration's variable-to-check messages, then every variable
% updates; the syndrome is checked after each iteration and decoding stops as
% soon as it is zero. messages = iterations * nnz(H).
%
% Numerics follow the sequential decoders here (sched_frame.m): |tanh| is
% floored at 1e-12 and the product capped at 1-1e-15, so messages stay finite
% at 50 iterations (an unclamped 2*atanh(1) is Inf).
%
% INPUTS  Lch (1 x n channel LLRs), env (sched_env), i_max
% OUTPUT  out.L (posterior), .converged, .iterations, .messages

er = env.er;  ec = env.ec;  m = env.m;  n = env.n;
v2c = Lch(ec)';                       % column, one entry per edge
L = Lch;  converged = false;  it_done = i_max;
for it = 1:i_max
    t = tanh(v2c / 2);
    small = abs(t) < 1e-12;
    t(small & t >= 0) = 1e-12;
    t(small & t < 0)  = -1e-12;
    la  = log(abs(t));
    neg = t < 0;
    S   = accumarray(er, la, [m 1]);
    par = mod(accumarray(er, double(neg), [m 1]), 2);
    sgn = 1 - 2 * xor(par(er) == 1, neg);
    p   = sgn .* exp(S(er) - la);
    p   = min(max(p, -(1 - 1e-15)), 1 - 1e-15);
    c2v = log((1 + p) ./ (1 - p));
    L   = Lch + accumarray(ec, c2v, [n 1])';
    v2c = L(ec)' - c2v;
    if all(mod(env.H * double(L <= 0)', 2) == 0)
        converged = true;  it_done = it;
        break;
    end
end
out = struct('L', L, 'converged', converged, 'iterations', it_done, ...
             'messages', it_done * env.nnzH, 'reward', 0);
end
