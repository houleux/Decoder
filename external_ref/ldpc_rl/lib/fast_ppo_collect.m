function part = fast_ppo_collect(env, net, posf, i_max, snrs, quota, bmax, seed, substream)
% FAST_PPO_COLLECT - collect at least `quota` PPO transitions with batched
% rollouts (fast_ppo_rollout.m), for one worker in one collection round.
%
% Noise and action sampling come from RandStream('Threefry') substream
% `substream` of seed (noise) and seed + 7919 (sampling), so a round's data
% depend only on (seed, substream), not on which worker ran it. Batch sizes
% shrink towards the end so the round overshoots `quota` by about one frame,
% as the one-frame-at-a-time trainer does. Training SNRs are cycled frame by
% frame within the round.
%
% OUTPUT part.ep (transitions, episodes contiguous, last step terminal),
%        part.episodes, part.reward (sum of per-frame rewards)

n = env.n;  nS = numel(snrs);
isOct = exist('OCTAVE_VERSION', 'builtin') ~= 0;
if isOct
    randn('state', seed * 1000003 + substream);  rand('state', (seed + 7919) * 1000003 + substream);
    noise = @(B) randn(B, n);  unif = @(B, c) rand(B, c);
else
    sn = RandStream('Threefry', 'Seed', seed);  sn.Substream = substream;
    su = RandStream('Threefry', 'Seed', seed + 7919);  su.Substream = substream;
    noise = @(B) randn(sn, B, n);  unif = @(B, c) rand(su, B, c);
end
Tm = env.K * i_max;
parts = {};  N = 0;  frames = 0;  reward = 0;  savg = Tm / 2;
while N < quota
    B = min(bmax, max(1, ceil((quota - N) / savg)));
    snr = snrs(mod(frames + (0:B-1), nS) + 1)';
    sigma = sqrt(1 ./ (2 * env.code_rate * 10.^(snr / 10)));
    Lch = 2 * (1 + sigma .* noise(B)) ./ sigma.^2;
    o = fast_ppo_rollout(Lch, env, net, posf, i_max, true, unif(B, Tm));
    parts{end + 1} = o.ep;                                     %#ok<AGROW>
    N = N + numel(o.ep.a);  frames = frames + B;  reward = reward + sum(o.reward);
    savg = N / frames;
end
part.ep = cat_eps(parts);
part.episodes = frames;  part.reward = reward;
end

function ep = cat_eps(parts)
ep = parts{1};
for i = 2:numel(parts)
    p = parts{i};
    ep.X = cat(3, ep.X, p.X);  ep.mask = [ep.mask, p.mask];
    ep.a = [ep.a, p.a];  ep.logp = [ep.logp, p.logp];  ep.v = [ep.v, p.v];
    ep.r = [ep.r, p.r];  ep.done = [ep.done, p.done];
end
end
