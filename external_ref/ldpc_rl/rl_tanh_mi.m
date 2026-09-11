%% ================= RELDEC OPTIMIZED GPU SCRIPT =================
clear; clc;

%% ---------------- PARITY CHECK MATRIX ----------------
% load("P_520.mat","P_520")
load('wran_384_256.mat','wran_384_256');
% load('p_mackey.mat','p_mackey');
% P = P_520;
% blocksize = 10;
% blocksize = 16;
H = sparse(logical(wran_384_256));
% H = ldpcQuasiCyclicMatrix(blocksize,P);
% H = sparse(logical(p_mackey));
blocksize = 1 ;
[m, ~] = size(H);

%% ---------------- PARAMETERS ----------------
params.alpha = 0.1;
params.beta = 0.9;
params.epsilon = 0.1;
params.lmax = 50;

params.maxlevels = 22;   % Quantization Level 32 -> 5 bits


numSamples = 90000;
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


snr = [1	1.25892541179417	1.58489319246111	1.99526231496888	2.51188643150958];
% snr = [0.501187233627272	0.630957344480193	0.794328234724282	1	1.25892541179417];
% snr = [1.122018454301963	1.258925411794167	1.412537544622754	1.584893192461114	1.778279410038923	1.995262314968880];
sigma = 1;

for i = 1:numSamples
    rx = 1*sqrt(snr(1)) + sigma * randn(1,n);
    L_set{i} = 2*rx/(sigma^2);
end

%% ---------------- TRAIN ----------------
Q = RELDEC_CPU_MAIN(L_set, H, CN_neighbors, VN_neighbors, clusters, params);


save("Q_wran_0_tanh_mi.mat","Q")
disp('Training completed');


% %% ================= MAIN FUNCTION =================
function Q = RELDEC_CPU_MAIN(L_set, H, CN_neighbors, VN_neighbors, clusters, params)

alpha   = params.alpha;
beta    = params.beta;
epsilon = params.epsilon;
lmax    = params.lmax;

[m, n] = size(H);
numClusters = length(clusters);
maxStates = params.maxlevels;

% Q = 0.01 * rand(maxStates, numClusters);   % avoid symmetry lock
Q = zeros(maxStates, numClusters);
N = length(L_set);
tic;

for idx = 1:N

    L = L_set{idx}(:)';   % column
    current_state = zeros(m, 1);
    s = zeros(m,1);
    for i = 1:m
        % Initial State
        idx1 = CN_neighbors{i};          % neighbor indices
        vals = L(idx1);                 % extract values

        % ensure row vector
        vals = vals(:)';

        current_state(i) = mean(tanh((vals))); 

        % Initilization of states for Episode
        s(i) = min(max(floor((current_state(i) + 1)/2 * maxStates) + 1, 1), maxStates);

    end
    
    
    % Creating Residue Vectors for BP Alogorithm
    for i = 1 : m
        res{i} = zeros(1,numel(CN_neighbors{i}));
    end
    
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
        %% -------- calculating current mutual inforamtion of the cluster --------
        idx2 = CN_neighbors{a};          
        x = L(idx2);
        mi_prev = mean(Jfun(x));
        %% -------- CN → VN (exact BP) --------
        idx2 = CN_neighbors{a};          
        vals2 = L(idx2)- res{a};
        temp = tanh(vals2./2);
        prodLq = prod(temp);  
        res{a} = 2*atanh(prodLq ./ temp);
        L(idx2) = vals2 + res{a};
        %% -------- calculating current mutual inforamtion of the cluster --------
        idx2 = CN_neighbors{a};          
        x = L(idx2);
        mi_new = mean(Jfun(x));
        %% -------- NEW STATE --------
        current_state_updated = zeros(m, 1);
        s_new = zeros(m,1);
        for i = 1:m
            idx1 = CN_neighbors{i};          % neighbor indices
            vals = L(idx1);                 % extract values

            % ensure row vector
            vals = vals(:)';
            
            current_state_updated(i) = mean(tanh((vals)));

            % Initilization of states for Episode
            s_new(i) = min(max(floor((current_state_updated(i) + 1)/2 * maxStates) + 1, 1), maxStates);
            
        end

        %% -------- REWARD --------
        reward = mi_new - mi_prev;
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