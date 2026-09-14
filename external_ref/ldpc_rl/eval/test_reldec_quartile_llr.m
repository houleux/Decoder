%% ===== EVAL — quartile-average-LLR RELDEC (k = 3, 4, 5) on WRAN =====
% Evaluates the checkpoints written by train/reldec_quartile_llr.m: for
% each k, runs the trained scheduling policy (pure greedy Q-table lookup,
% epsilon = 0 — no exploration at eval time) as a check-node scheduler
% for exact-BP LDPC decoding, over an SNR sweep, for numFrames frames per
% SNR point, and reports BER/FER. Plots all three k curves together.
%
% Unlike eval/test_reldec.m, this harness does not use the Communications
% Toolbox (no ldpcEncode/ldpcEncoderConfig): it generates the all-zero
% codeword directly as BPSK+AWGN channel LLRs over all n bits, exactly as
% train/reldec_quartile_llr.m's own L_set generator does, and reports BER
% over all n bits (this matrix has no separate encoder config in this
% repo to isolate systematic/information bits). Decoding runs for
% maxIter full "sweeps" worth of scheduling budget (m actions per sweep)
% with early stopping the moment the syndrome is all-zero.
%
% Run from external_ref/ldpc_rl/eval/ (or addpath(genpath('..')) first).
clear; clc;
addpath('../lib');

%% ---------------- PARITY CHECK MATRIX ----------------
load('../matrices/wran_384_256.mat', 'wran_384_256');
H = sparse(logical(wran_384_256));
[m, n] = size(H);
R = 1 - m/n;

%% ---------------- EVAL CONFIG ----------------
SNR_db    = [0 1 2 3 4];
SNR       = 10.^(SNR_db/10);
numFrames = 1000;
maxIter   = 5;          % scheduling budget = m * maxIter actions per frame
k_vals    = [3, 4, 5];

%% ---------------- PRECOMPUTE GRAPH ----------------
CN_neighbors = cell(m,1);
for c = 1:m
    CN_neighbors{c} = find(H(c,:));
end
deg_groups = build_deg_groups(CN_neighbors);   % for quartile_state_batch.m
cols = (1:m)';

%% ---------------- LOAD CHECKPOINTS ----------------
Qtabs = cell(1, length(k_vals));
for kk = 1:length(k_vals)
    d = load(sprintf('../checkpoints/Q_wran_quartile_k%d.mat', k_vals(kk)), 'Q');
    Qtabs{kk} = d.Q;
end

%% ---------------- RUN ----------------
ber_t = zeros(length(k_vals), length(SNR_db));
fer_t = zeros(length(k_vals), length(SNR_db));

for kk = 1:length(k_vals)
    k = k_vals(kk);
    Q = Qtabs{kk};
    pow2vec = 2.^(k-1:-1:0);

    for si = 1:length(SNR_db)
        sigma = sqrt(1/(2*R*SNR(si)));
        bit_errors = 0;
        frame_errors = 0;

        for f = 1:numFrames
            rx = 1 + sigma*randn(1,n);
            L = 2*rx/(sigma^2);

            res = cell(m,1);
            for i = 1:m
                res{i} = zeros(1, numel(CN_neighbors{i}));
            end

            state_hard = quartile_state_batch(L, deg_groups, k, m) < 0;

            for step = 1:(m*maxIter)
                % ---- syndrome-based early stop ----
                hard = double(L < 0)';
                if all(mod(H*hard, 2) == 0)
                    break;
                end

                % ---- greedy action (epsilon = 0 at eval) ----
                s = 1 + state_hard * pow2vec';
                vals1 = Q(sub2ind(size(Q), s, cols))';
                [~, a] = max(vals1);

                % ---- CN -> VN (exact BP) ----
                idx2 = CN_neighbors{a};
                vals2 = L(idx2) - res{a};
                temp = tanh(vals2./2);
                prodLq = prod(temp);
                res{a} = 2*atanh(prodLq ./ temp);
                L(idx2) = vals2 + res{a};

                % ---- new state ----
                state_hard = quartile_state_batch(L, deg_groups, k, m) < 0;
            end

            hard_final = (L < 0);
            bit_errors = bit_errors + nnz(hard_final);          % transmitted codeword is all-zero
            frame_errors = frame_errors + double(any(hard_final));

            if mod(f, 200) == 0
                fprintf('k=%d SNR_db=%d frame %d/%d\n', k, SNR_db(si), f, numFrames);
            end
        end

        ber_t(kk, si) = bit_errors / (numFrames * n);
        fer_t(kk, si) = frame_errors / numFrames;
        fprintf('k=%d SNR_db=%d  BER=%.6g  FER=%.6g\n', k, SNR_db(si), ber_t(kk,si), fer_t(kk,si));
    end
end

%% ---------------- REPORT ----------------
fprintf('\n===== Summary (k x SNR_db) =====\n');
fprintf('%8s', 'k\\SNRdB');
for si = 1:length(SNR_db)
    fprintf('%12g', SNR_db(si));
end
fprintf('\n');
for kk = 1:length(k_vals)
    fprintf('BER k=%d', k_vals(kk));
    for si = 1:length(SNR_db)
        fprintf('%12.4g', ber_t(kk,si));
    end
    fprintf('\n');
end
for kk = 1:length(k_vals)
    fprintf('FER k=%d', k_vals(kk));
    for si = 1:length(SNR_db)
        fprintf('%12.4g', fer_t(kk,si));
    end
    fprintf('\n');
end

save('../checkpoints/quartile_llr_eval_results.mat', 'SNR_db', 'k_vals', 'ber_t', 'fer_t');

%% ---------------- PLOT ----------------
figure;
markers = {'-o', '-s', '-^'};
for kk = 1:length(k_vals)
    semilogy(SNR_db, ber_t(kk,:), markers{kk}, 'DisplayName', sprintf('k = %d', k_vals(kk)));
    hold on;
end
grid on;
xlabel('SNR (dB)');
ylabel('BER');
title('Quartile-average-LLR RELDEC — WRAN 384x256');
legend('show');
saveas(gcf, '../checkpoints/quartile_llr_ber.png');
