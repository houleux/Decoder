function fast_ppo_smoketest()
% FAST_PPO_SMOKETEST - fast_ppo_rollout.m must reproduce sched_frame.m's 'ppo'
% kind frame by frame, in both modes:
%   greedy   identical converged / iterations / messages, LLRs equal to rounding
%   training identical actions, rewards, done flags and episode lengths when
%            frame b's sampling draws are the same; log-probs, values and
%            observations equal to rounding
% on WRAN z = 4 and 16, a random and a trained policy, with and without the
% position feature, 1-4 dB. Errors out on any mismatch.
here = fileparts(mfilename('fullpath'));
ck = fullfile(here, '..', 'checkpoints', 'sched');
tol = 1e-9;
for z = [4 16]
    env = sched_env('wran', z);
    K = env.K;  n = env.n;  i_max = 5;  Tm = K * i_max;
    nets = {};
    reseed(1);  nets{end + 1} = struct('net', ppo_init(9, 32), 'posf', false, 'name', 'random');
    reseed(2);  nets{end + 1} = struct('net', ppo_init(10, 32), 'posf', true, 'name', 'random+pos');
    f = fullfile(ck, sprintf('wran_ppo_z%d_ep100000.mat', z));
    if exist(f, 'file')
        c = load(f);  nets{end + 1} = struct('net', c.pol.net, 'posf', c.pol.pos_feature, 'name', 'trained 100k');
    end
    for q = 1:numel(nets)
        net = nets{q}.net;  posf = nets{q}.posf;
        B = 24;  snr = repmat([1 2 3 4], 1, B / 4)';
        sigma = sqrt(1 ./ (2 * env.code_rate * 10.^(snr / 10)));
        reseed(100 + z + q);
        Lch = 2 * (1 + sigma .* randn(B, n)) ./ sigma.^2;
        pol = struct('kind', 'ppo', 'net', net, 'pos_feature', posf, 'reward', 'success');
        % greedy
        o = fast_ppo_rollout(Lch, env, net, posf, i_max, false, []);
        dL = 0;
        for b = 1:B
            s = sched_frame(Lch(b, :), env, pol, struct('i_max', i_max, 'train', false));
            chk(s.converged == o.converged(b) && s.iterations == o.iterations(b) && s.messages == o.messages(b), ...
                'greedy z=%d %s frame %d: decode outcome differs', z, nets{q}.name, b);
            dL = max(dL, max(abs(s.L - o.L(b, :)) ./ max(1, abs(s.L))));
        end
        chk(dL < tol, 'greedy z=%d %s: LLR mismatch %g', z, nets{q}.name, dL);
        ngd = nnz(o.converged);
        % training: frame b's draws are U(b, :)
        U = zeros(B, Tm);
        for b = 1:B, reseed(5000 + b);  U(b, :) = rand(1, Tm); end
        o = fast_ppo_rollout(Lch, env, net, posf, i_max, true, U);
        off = 0;  dX = 0;  dLP = 0;  dV = 0;
        for b = 1:B
            reseed(5000 + b);
            s = sched_frame(Lch(b, :), env, pol, struct('i_max', i_max, 'train', true));
            T = numel(s.ep.a);  id = off + (1:T);  off = off + T;
            chk(T == o.T(b) && isequal(s.ep.a, o.ep.a(id)) && isequal(s.ep.r, o.ep.r(id)) && ...
                isequal(s.ep.done, o.ep.done(id)) && isequal(s.ep.mask, o.ep.mask(:, id)) && ...
                s.converged == o.converged(b), 'training z=%d %s frame %d: trajectory differs', z, nets{q}.name, b);
            dX = max(dX, max(abs(double(s.ep.X(:)) - double(reshape(o.ep.X(:, :, id), [], 1)))));
            dLP = max(dLP, max(abs(s.ep.logp - o.ep.logp(id))));
            dV = max(dV, max(abs(s.ep.v - o.ep.v(id))));
        end
        chk(off == numel(o.ep.a), 'training z=%d: %d extra transitions', z, numel(o.ep.a) - off);
        chk(dX < 1e-6 && dLP < tol && dV < tol, 'training z=%d %s: |dX| %g |dlogp| %g |dv| %g', z, nets{q}.name, dX, dLP, dV);
        fprintf('fast_ppo_smoketest z=%-2d %-13s greedy: %2d/%d decoded, max rel dL %.1e | training: %4d steps, %2d/%d converged, max |dX| %.1e |dlogp| %.1e |dv| %.1e  OK\n', ...
            z, nets{q}.name, ngd, B, dL, off, nnz(o.converged), B, dX, dLP, dV);
    end
end
fprintf('fast_ppo_smoketest: ALL PASSED\n');
end

function reseed(s)
if exist('OCTAVE_VERSION', 'builtin'), rand('state', s);  randn('state', s);
else, RandStream.setGlobalStream(RandStream('Threefry', 'Seed', s)); end
end

function chk(ok, varargin)
if ~ok, error('fast_ppo_smoketest:mismatch', varargin{:}); end
end
