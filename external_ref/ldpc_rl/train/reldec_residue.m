%% ================= RELDEC OPTIMIZED GPU SCRIPT =================
clear; clc;

%% ---------------- PARITY CHECK MATRIX ----------------
% load("P_520_100.mat","P_520_100")
load('wran_384_256.mat','wran_384_256');
% P = P_520_100;
% blocksize = 10;
H = sparse(logical(wran_384_256));
% H = ldpcQuasiCyclicMatrix(blocksize,P);
[m, ~] = size(H);

%% ---------------- PARAMETERS ----------------
params.alpha = 0.1;
params.beta = 0.9;
params.epsilon = 0.1;
params.lmax = 50;
params.maxStateBits = 11;   % IMPORTANT 10 for P_520

numSamples = 100000;
n =length(H);
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
SNR_db = 0;
snr = 10.^(SNR_db/10);
sigma = 1;

for i = 1:numSamples
    rx = 1*sqrt(snr) + sigma * randn(1,n);
    L_set{i} = 2*rx/(sigma^2);
end

%% ---------------- TRAIN ----------------
Q = RELDEC_CPU_MAIN(L_set, H, CN_neighbors, VN_neighbors, clusters, params);
save("Q_wran_snr_0_new_reward","Q")
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
for i = 1 : 4
Q{i} = zeros(maxStates, numClusters);
end
N = length(L_set);
tic;

for idx = 1:N

    L = L_set{idx}(:)';   % column
    for i = 1 : m
        res{i} = zeros(1,numel(CN_neighbors{i}));
    end
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
        
        %Calculating Residue for current state
        w = min(abs(vals(1:k)-res{i}));
        if w < 0.25
            q = 1;
        elseif w < 0.5
            q = 2;
        elseif w < 0.75
            q = 3;
        else
            q = 4;
        end
        res_quant(i) = q;
    end
 
    
    %  Hard Decoded State
    state_hard = zeros(m,params.maxStateBits);
    for i = 1 : m
        state_hard(i,:) = current_state(i, :) < 0;
    end
    % Creating Residue Vectors for BP Alogorithm
    
    % Initilization of states for Episode
    % bin2dec for Q indexing
    s = zeros(1,m);
    for i = 1 : m
        vec = state_hard(i,:);
        s(i) = 1 + sum(vec .* (2.^(length(vec)-1:-1:0)));
    end
    % Possible Actions in the current state
    vals1 = zeros(1, m);

    for i = 1:m
        vals1(i) = Q{res_quant(i)}(s(i), i);
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
        idx2 = CN_neighbors{a};          
        vals2 = L(idx2)- res{a};
        temp = tanh(vals2./2);
        prodLq = prod(temp);
        res_old = res{a};
        res{a} = 2*atanh(prodLq ./ temp);
        L_new(idx2) = vals2 + res{a};
        %% -------- NEW STATE --------
        current_state_updated = zeros(m, params.maxStateBits);
        for i = 1:m
            idx1 = CN_neighbors{i};          % neighbor indices
            vals = L_new(idx1);                 % extract values

            % ensure row vector
            vals = vals(:)';

            k = min(length(vals), params.maxStateBits);

            % copy first k elements
            current_state_updated(i,1:k) = vals(1:k);


            % remaining elements already zero (no need to fill)
            % Calculating the updated residue
            w = min(abs(vals(1:k)-res{i}));
            if w < 0.25
                q = 1;
            elseif w < 0.5
                q = 2;
            elseif w < 0.75
                q = 3;
            else
                q = 4;
            end
            res_quant_updated(i) = q;

        end
        %  Hard Decoded State
        state_hard_updated = zeros(m,params.maxStateBits);
        for i = 1 : m
            state_hard_updated(i,:) = current_state_updated(i, :) < 0;
        end

        %% -------- REWARD --------
        % prev_correct_bits = sum(state_hard(a,:));
        % new_correct_bits = sum(state_hard_updated(a,:));
        % delta = sum(abs(current_state_updated(a,:)-current_state_updated(a,:)));
        delta = sum(abs(res{a}-res_old));
        % 
        % reward = ((new_correct_bits - prev_correct_bits)+delta)/(length(CN_neighbors{a}));
        reward = (sum(state_hard_updated(a,:))+delta)/(length(CN_neighbors{a}));
        % bin2dec for Q indexing fr updated state
        for i = 1 : m
            vec = state_hard_updated(i,:);
            s_new(i) = 1 + sum(vec .* (2.^(length(vec)-1:-1:0)));
        end
        % Possible Actions in the current state
        for i = 1:m
            vals1(i) = Q{res_quant_updated(i)}(s_new(i), i);
        end
        %% -------- Q UPDATE --------
        Q{res_quant(a)}(s(a),a) = (1-alpha)*Q{res_quant(a)}(s(a),a) + ...
                 alpha*(reward + beta*max(vals1));

        % update state
        state_hard = state_hard_updated;
        res_quant = res_quant_updated;
        L = L_new;
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