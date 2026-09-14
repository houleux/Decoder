%% ================= RELDEC OPTIMIZED GPU SCRIPT =================
clear; clc;

%% ---------------- PARITY CHECK MATRIX ----------------
% load("P_520.mat","P_520")
load('wran_384_256.mat','wran_384_256');
% load('p_mackey.mat','p_mackey');
% P = P_520;
blocksize = 10;
% blocksize = 16;
H = sparse(logical(wran_384_256));
% H = ldpcQuasiCyclicMatrix(blocksize,P);
% H = sparse(logical(p_mackey));
% blocksize = 1 ;
[m, ~] = size(H);
pcmatrix = H;
cfgLDPCEnc = ldpcEncoderConfig(pcmatrix);

%% ---------------- PARAMETERS ----------------
params.alpha = 0.1;
params.beta = 0.9;
params.epsilon = 0.1;
params.lmax = 50;

params.maxStateBits = 11;   % IMPORTANT   10 for P_520 and 11 for WRAN 6 for mackey


numSamples = 30000;
n = length(H);
%% ---------------- PRECOMPUTE GRAPH ----------------
CN_neighbors = cell(m,1);
VN_neighbors = cell(n,1);

for c = 1:m
    CN_neighbors{c} = find(H(c,:));
end

for v = 1:n
    VN_neighbors{v} = find(H(:,v));
end

%% ---------------- CLUSTERS ----------------
clusters = num2cell(1:m);

%% ---------------- GENERATE TRAINING DATA ----------------
L_set = cell(numSamples,1);


snr = [1	1.25892541179417	1.58489319246111	1.99526231496888	2.51188643150958 	3.16227766016838	3.98107170553497];
% snr = [0.501187233627272	0.630957344480193	0.794328234724282	1	1.25892541179417];
% snr = [1.122018454301963	1.258925411794167	1.412537544622754	1.584893192461114	1.778279410038923	1.995262314968880];
% sigma = 1;
R = cfgLDPCEnc.NumInformationBits / cfgLDPCEnc.BlockLength;
% R = 1/2 % for mackey
sigma = sqrt(1/(2*R*snr(7)));
for i = 1:numSamples
    rx = 1 +  sigma*randn(1,n);
    L_set{i} = 2*rx/(sigma^2);
end

%% ---------------- TRAIN ----------------
Q = RELDEC_CPU_MAIN(L_set, H, CN_neighbors, VN_neighbors, clusters, params);


save("Q_wran_crt_llr_6.mat","Q")
disp('Training completed');


% %% ================= MAIN FUNCTION =================
function Q = RELDEC_CPU_MAIN(L_set, H, CN_neighbors, VN_neighbors, clusters, params)

alpha   = params.alpha;
beta    = params.beta;
epsilon = params.epsilon;
lmax    = params.lmax;

[m, n] = size(H);
numClusters = length(clusters);
maxStates = 2^params.maxStateBits;

% Q = 0.01 * rand(maxStates, numClusters);   % avoid symmetry lock
Q = zeros(maxStates, numClusters);
N = length(L_set);
tic;

for idx = 1:N

    L = L_set{idx}(:)';   % column
    current_state = zeros(m, params.maxStateBits);
    for i = 1:m
        % Initial State
        idx1 = CN_neighbors{i};          % neighbor indices
        vals = L(idx1);                 % extract values

        % ensure row vector
        vals = vals(:)';

        k = min(length(vals), params.maxStateBits);

        % copy first k elements
        current_state(i,1:k) = vals(1:k);

        % remaining elements already zero (no need to fill)

    end
    %  Hard Decoded State
    state_hard = current_state < 0;
    % Creating Residue Vectors for BP Alogorithm
    for i = 1 : m
        res{i} = zeros(1,numel(CN_neighbors{i}));
    end
    % Initilization of states for Episode
    % bin2dec for Q indexing
    pow2vec = 2.^(params.maxStateBits-1:-1:0);
    s = 1 + state_hard * pow2vec';
    % Possible Actions in the current state
    vals1 = zeros(1, m);

    for i = 1:m
        vals1(i) = Q(s(i), i);
    end
    % start of an episode
    for l = 1:lmax
        %% -------- ACTION --------
        if rand < epsilon
            a = randi(numClusters);
        else
            [~, a] = max(vals1);
        end
        %% -------- CN → VN (exact BP) --------
        % res_prev = zeros(1,numel(CN_neighbors{a}));
        idx2 = CN_neighbors{a};          
        vals2 = L(idx2)- res{a};
        temp = tanh(vals2./2);
        prodLq = prod(temp);
        res_prev = res{a};
        res{a} = 2*atanh(prodLq ./ temp);
        L(idx2) = vals2 + res{a};
        %% -------- NEW STATE --------
        current_state_updated = zeros(m, params.maxStateBits);
        for i = 1:m
            idx1 = CN_neighbors{i};          % neighbor indices
            vals = L(idx1);                 % extract values

            % ensure row vector
            vals = vals(:)';

            k = min(length(vals), params.maxStateBits);

            % copy first k elements
            current_state_updated(i,1:k) = vals(1:k);

            % remaining elements already zero (no need to fill)

        end
        %  Hard Decoded State
        state_hard_updated = current_state_updated < 0;

        %% -------- REWARD --------
        % prev_correct_bits = sum(state_hard(a,:));
        % new_correct_bits = sum(state_hard_updated(a,:));
        % 
        % reward = prev_correct_bits - new_correct_bits;
        % reward = nnz(state_hard_updated(a,:) == 0)/(length(CN_neighbors{a}));
        % reward = sum(abs(res{a}-res_prev));
        temp_llr = zeros(1,11);
        k2 = length(L(idx2));
        temp_llr(1:k2) = L(idx2);
        reward = sum((state_hard_updated(a,:) == 0).*temp_llr)/(length(CN_neighbors{a}));
        % bin2dec for Q indexing fr updated state
        s_new = 1 + state_hard_updated*pow2vec';
        % Possible Actions in the current state
        for i = 1:m
            vals1(i) = Q(s_new(i), i);
        end
        %% -------- Q UPDATE --------
        Q(s(a),a) = (1-alpha)*Q(s(a),a) + ...
                 alpha*(reward + beta*max(vals1));

        % State update for next iteration
        s = s_new;
    end

    %% -------- PROGRESS --------
    if mod(idx,1000) == 0
        elapsed = toc;
        rate = idx / elapsed;
        remaining = (N - idx) / rate;

        fprintf('Episode %d/%d (%.2f%%) | ETA: %.1fs\n', ...
            idx, N, 100*idx/N, remaining);
    end

end

end