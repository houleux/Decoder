%% ========= RELDEC — QUARTILE-AVERAGE-LLR STATE (parameterized by k) =========
% New architecture (per user request, 2026-09-11). Same factored-MDP /
% tabular-Q-learning skeleton as train/reldec_test.m (one cluster == one
% check node, epsilon-greedy action selection, exact-BP CN->VN update as
% the environment step) but with a different state encoder:
%
%   State for check node i = a vector of size k. Sort the LLRs of check
%   node i's neighboring variable nodes ascending, split them into k
%   groups as evenly as possible, and take the average LLR within each
%   group. (k = 4 is "quartiles"; k is a parameter — see lib/quartile_state.m.)
%
% The vector is then hard-decided (< 0) and bin2dec-packed into a
% 2^k-row Q-table index, exactly like reldec_test.m packs its
% maxStateBits raw neighbor LLRs. Reward is the reldec_test.m formula,
% substituting k for maxStateBits: the fraction of the cluster's k
% quantile-group hard bits that are "satisfied" (>= 0) after the update,
% corrected for any zero-padded groups (see lib/quartile_state.m; for the
% WRAN matrix, row degree 10-11 always exceeds k in {3,4,5}, so the
% correction term is always 0 in practice).
%
% Trains k = 3, 4, 5 back to back on the WRAN matrix and saves one
% checkpoint per k. See docs/notes/quartile_llr_reldec.md for design
% rationale and docs/notes/quartile_llr_reldec_results.md for results.
%
% Run from external_ref/ldpc_rl/train/ (or addpath(genpath('..')) first,
% since this script and quartile_state.m live in different directories).
clear; clc;
addpath('../lib');

%% ---------------- PARITY CHECK MATRIX ----------------
load('../matrices/wran_384_256.mat', 'wran_384_256');
H = sparse(logical(wran_384_256));
[m, n] = size(H);
R = 1 - m/n;   % code rate, computed directly from H (reldec_test.m left
               % this variable undefined — see docs/notes/quartile_llr_reldec.md)

%% ---------------- PARAMETERS ----------------
params.alpha   = 0.1;
params.beta    = 0.9;
params.epsilon = 0.1;
params.lmax    = 50;

numSamples = 30000;

%% ---------------- PRECOMPUTE GRAPH ----------------
CN_neighbors = cell(m,1);
for c = 1:m
    CN_neighbors{c} = find(H(c,:));
end
deg_groups = build_deg_groups(CN_neighbors);   % for quartile_state_batch.m

%% ---------------- CLUSTERS ----------------
clusters = num2cell(1:m);

%% ---------------- GENERATE TRAINING DATA ----------------
% Same all-ones-codeword AWGN generator as reldec_test.m, fixed at the
% first (highest-SNR) point of its snr list.
L_set = cell(numSamples,1);
snr = [1 1.25892541179417 1.58489319246111 1.99526231496888 2.51188643150958 3.16227766016838 3.98107170553497];
sigma = sqrt(1/(2*R*snr(1)));

for i = 1:numSamples
    rx = 1 + sigma*randn(1,n);
    L_set{i} = 2*rx/(sigma^2);
end

%% ---------------- TRAIN FOR EACH k ----------------
k_vals = [3, 4, 5];
for kk = 1:length(k_vals)
    params.k = k_vals(kk);
    fprintf('=== Training quartile-LLR RELDEC, k = %d ===\n', params.k);
    Q = RELDEC_QUARTILE_MAIN(L_set, CN_neighbors, deg_groups, clusters, params);
    outfile = sprintf('../checkpoints/Q_wran_quartile_k%d.mat', params.k);
    save(outfile, 'Q', 'params');
    fprintf('Saved %s\n', outfile);
end
disp('Training completed for k = 3, 4, 5');

