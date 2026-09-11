%% ================= RELDEC OPTIMIZED GPU SCRIPT =================
clear; clc;

%% ---------------- PARITY CHECK MATRIX ----------------
% load("P_520.mat","P_520")
% P = P_520;
P = [  0   1   1   0  -1  -1   2  -1  -1   1   0   0  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1 
  0  -1  -1   0   0   2   1   1   2   0  -1   0   0  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1 
  1   0  -1   0   2  -1  -1  -1   0  -1   1  -1   0   0  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1 
 -1   1   1  -1   2   0   0   1   1   0   0  -1  -1   0  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1 
  0   2  -1  -1  -1  -1  -1  -1  -1  -1  -1   2  -1  -1   0  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1 
  1   2  -1  -1  -1   1  -1   2  -1  -1  -1   0  -1  -1  -1   0  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1 
  0  -1  -1  -1  -1   2  -1   1  -1   1  -1   1  -1  -1  -1  -1   0  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1 
 -1   2  -1  -1  -1   0  -1   1  -1  -1  -1   0  -1   2  -1  -1  -1   0  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1 
  1   1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1   2  -1  -1  -1  -1  -1   0  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1 
 -1   1  -1  -1  -1  -1  -1  -1   0  -1   2   1  -1  -1  -1  -1  -1  -1  -1   0  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1 
  2   2  -1  -1  -1  -1   1   1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1   0  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1 
  2  -1  -1  -1  -1  -1  -1   2  -1   0  -1  -1  -1   1  -1  -1  -1  -1  -1  -1  -1   0  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1 
 -1   0  -1   0  -1  -1  -1  -1  -1  -1  -1   2  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1   0  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1 
  1   2  -1  -1  -1  -1  -1  -1   2  -1  -1  -1  -1   1  -1  -1  -1  -1  -1  -1  -1  -1  -1   0  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1 
 -1   1  -1  -1  -1  -1   1  -1  -1  -1  -1   0  -1   1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1   0  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1 
  2  -1  -1  -1  -1  -1  -1  -1  -1  -1   0   0  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1   0  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1 
 -1   0  -1  -1  -1  -1  -1  -1  -1   0  -1   0   1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1   0  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1 
 -1   2  -1  -1  -1   2  -1  -1  -1  -1  -1   0   0  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1   0  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1 
  0  -1  -1  -1  -1  -1   0   1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1   0  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1 
  1   0  -1  -1  -1  -1  -1  -1  -1  -1   0  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1   0  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1 
 -1   1  -1  -1   1  -1  -1  -1  -1  -1  -1   0  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1   0  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1 
  1  -1  -1  -1  -1  -1  -1  -1   1  -1  -1  -1  -1   1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1   0  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1 
 -1   2   1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1   0  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1 
  1  -1  -1   2  -1   2  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1   0  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1 
 -1   2   0  -1  -1  -1  -1  -1  -1   1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1   0  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1 
  0  -1  -1  -1  -1   0  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1   0  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1 
 -1  -1   1  -1  -1  -1  -1   2  -1  -1  -1  -1   1   1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1   0  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1 
  1  -1  -1  -1  -1  -1   2  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1   0  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1 
 -1   1   0  -1  -1   0  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1   0  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1 
  2  -1  -1  -1   2  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1   0  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1 
 -1  -1   0  -1  -1   1  -1   1  -1   2  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1   0  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1 
 -1   0  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1   2  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1   0  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1 
  0  -1  -1  -1  -1   2  -1  -1  -1  -1  -1  -1   2  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1   0  -1  -1  -1  -1  -1  -1  -1  -1  -1 
 -1  -1   0  -1  -1  -1  -1   2  -1  -1   1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1   0  -1  -1  -1  -1  -1  -1  -1  -1 
  2  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1   0   0  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1   0  -1  -1  -1  -1  -1  -1  -1 
 -1   2  -1  -1  -1   1  -1  -1  -1  -1  -1   0  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1   0  -1  -1  -1  -1  -1  -1 
  1  -1   1  -1  -1  -1  -1   2  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1   0  -1  -1  -1  -1  -1 
 -1  -1  -1  -1  -1  -1  -1  -1  -1  -1   1  -1  -1   1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1   0  -1  -1  -1  -1 
 -1   0  -1  -1  -1   1  -1  -1  -1  -1  -1   1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1   0  -1  -1  -1 
  0  -1  -1  -1  -1  -1  -1   0  -1  -1  -1  -1   0  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1   0  -1  -1 
 -1  -1   1  -1  -1  -1  -1  -1  -1  -1   2  -1  -1   1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1   0  -1 
 -1   0  -1  -1  -1   1  -1  -1  -1  -1  -1   0  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1   0 ];
blocksize = 3;
H = ldpcQuasiCyclicMatrix(3,P);
% load("AB_LDPC_3_7_Z4_196.mat",'H')
% H = sparse(logical(H));
cfgLDPCEnc = ldpcEncoderConfig(H);
[m, n] = size(H);

%% ---------------- PARAMETERS ----------------
params.alpha = 0.1;
params.beta = 0.9;
params.epsilon = 0.3;
params.lmax = 50;
params.maxStateBits = 10;   % IMPORTANT

numSamples = 150000;
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

snr = 0;
sigma = 1;

for i = 1:numSamples
    bits = randi([0 1], cfgLDPCEnc.NumInformationBits,1);
    codeword = ldpcEncode(bits,cfgLDPCEnc);
    rx = codeword + sigma * randn(n,1);
    L_set{i} = 2*rx/(sigma^2);
end

%% ---------------- TRAIN ----------------
Q = RELDEC_CPU_MAIN(L_set, H, CN_neighbors, VN_neighbors, blocksize, params);
save("Q_1e5_P_520","Q")
disp('Training completed');


% %% ================= MAIN FUNCTION =================
function Q = RELDEC_CPU_MAIN(L_set, H, CN_neighbors, VN_neighbors, blocksize, params)

alpha   = params.alpha;
beta    = params.beta;
epsilon = params.epsilon;
lmax    = params.lmax;

[m, n] = size(H);
numClusters = m/blocksize;
maxStates = 4^blocksize;    % 4 = number of levels of quantization

% Q = 0.01 * rand(maxStates, numClusters);   % avoid symmetry lock
Q = zeros(maxStates, numClusters);
N = length(L_set);
tic;

for idx = 1:1

    L = L_set{idx}(:)';   % column
    for i = 1:numClusters
        for j = 1 : blocksize
        % Initial State
        idx1 = CN_neighbors{(i-1)*blocksize+j};          % neighbor indices
        vals = sum(L(idx1));                      % state value
        % quantization
        dc = length(idx1);
        x = vals / dc;   % normalize
        % T = 2*sqrt(dc)/dc;         % Threshold for quantisation
        T = 0.5;
        if x < -T
            q = 0;
        elseif x < 0
            q = 1;
        elseif x < T
            q = 2;
        else
            q = 3;
        end
        current_state(i,j) = q;
        end
    end
    state_hard = current_state;
    % Creating Residue Vectors for BP Alogorithm
    for i = 1 : numClusters
        res{i} = zeros(n,1);
    end
    % Initilization of states for Episode
    % bin2dec for Q indexing
    s = zeros(1,numClusters);
    for i = 1 : numClusters
        vec = state_hard(i,:);
        s(i) = 1 + sum(vec .* (4.^(length(vec)-1:-1:0)));
    end
    % Possible Actions in the current state
    vals1 = zeros(1, numClusters);

    for i = 1:m/blocksize
        vals1(i) = Q(s(i), i);
    end
    % start of an episode
    for l = 1:lmax
        L_new = L;
        %% -------- ACTION --------
        if rand < epsilon
            a = randi(numClusters);
        else
            [~, a] = max(vals1);
        end
        %% -------- CN → VN (exact BP) --------
        LLR_in = L;
        Res_int = zeros(length(LLR_in),1);
        LLR_int = LLR_in;
        LLR_int = LLR_int - res{a};
        for j = 1 : blocksize
            idx1 = CN_neighbors{(a-1)*blocksize+j};
            llr_temp = LLR_int(idx1);
            temp = tanh(llr_temp./2);
            prodLq = prod(temp);
            Res_int(idx1) = 2*atanh(prodLq ./ temp);
        end
        LLR_out = LLR_int + Res_int;
        %% -------- NEW STATE --------
        for i = 1:numClusters
            for j = 1 : blocksize
                % Initial State
                idx1 = CN_neighbors{(i-1)*blocksize+j};          % neighbor indices
                vals = sum(L(idx1));                      % state value
                % quantization
                dc = length(idx1);
                x = vals / dc;   % normalize
                % T = 2*sqrt(dc)/dc;         % Threshold for quantisation
                T = 0.5;
                if x < -T
                    q = 0;
                elseif x < 0
                    q = 1;
                elseif x < T
                    q = 2;
                else
                    q = 3;
                end
                current_state_updated(i,j) = q;
            end
        end
        state_hard_updated = current_state_updated;
        %% -------- REWARD --------
        % prev_correct_bits = sum(state_hard(a,:));
        % new_correct_bits = sum(state_hard_updated(a,:));
        % 
        % reward = prev_correct_bits - new_correct_bits;
        reward = max(abs(Res_int-res{a}))
        reward = min(reward, 10);
        % bin2dec for Q indexing fr updated state
        for i = 1 : numClusters
            vec = state_hard_updated(i,:);
            s_new(i) = 1 + sum(vec .* (4.^(length(vec)-1:-1:0)));
        end
        % Possible Actions in the current state
        for i = 1:numClusters
            vals1(i) = Q(s_new(i), i);
        end
        %% -------- Q UPDATE --------
        Q(s(a),a) = (1-alpha)*Q(s(a),a) + ...
                 alpha*(reward + beta*max(vals1));

        % update state
        res{a} = Res_int;
        state_hard = state_hard_updated;
        L = LLR_out;
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