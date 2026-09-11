% Configuration
load("P_156.mat","P")
load("Q_RL.mat","Q")
% Q = Q ./ max(abs(Q(:)));
% P = P_520_100;
BlockSize = 3;
epsilon_test = 0.05;
pcmatrix = ldpcQuasiCyclicMatrix(BlockSize,P);
% Q = readmatrix('qtable_ep015000.csv');
% Q = Q(2:end, :);
% pcmatrix = sparse(logical(readmatrix('WRAN_irreg_384_256 (1).csv')));
%% 
[m, ~] = size(pcmatrix);
CN_neighbors = cell(m,1);
for c = 1:m
    CN_neighbors{c} = find(pcmatrix(c,:));
end
numClusters = m/BlockSize;
maxStates = 4^BlockSize;    % 4 = number of levels of quantization
cfgLDPCEnc = ldpcEncoderConfig(pcmatrix);
cfgLDPCDec_bp = ldpcDecoderConfig(pcmatrix);
cfgLDPCDec = ldpcDecoderConfig(pcmatrix,'layered-bp');
% Parameters for Dec
numEdges      = length(cfgLDPCDec.derivedParams.columnIndexMap)/2;
blockLen      = cfgLDPCDec.BlockLength;
parityLen     = cfgLDPCDec.NumParityCheckBits;
nRowsPerLayer = cfgLDPCDec.NumRowsPerLayer;
oWeight       = cfgLDPCDec.derivedParams.offsetWeight;
cIndexMap     = cfgLDPCDec.derivedParams.columnIndexMap;
algChoice     = cfgLDPCDec.AlgorithmChoice;
p_sub = pcmatrix((1-1)*(BlockSize)+1:1*BlockSize,:);
N = cIndexMap(1:length(cIndexMap)/2)' + 1;
rowOffset = oWeight(1:parityLen,1);
rowWeight = oWeight(parityLen + (1:parityLen),1);
columnIndex = cIndexMap(1:numEdges) + 1;
row_weight = sum((P+1) ~= 0, 2);
SNR_db = [-3 -2 -1 0 1];
SNR = 10.^(SNR_db/10);
maxnumiter = 5;
for i =1 : length(SNR)
    for j = 1 : 1
        bits = zeros(cfgLDPCEnc.NumInformationBits,1);
        codeword = ldpcEncode(bits,cfgLDPCEnc);
        codeword_1 = (codeword == 0);
        codeword_2 = (codeword == 1);
        data_modulated = sqrt(SNR(i)).*codeword_1 -sqrt(SNR(i)).*codeword_2;
        noise = randn(cfgLDPCEnc.BlockLength,1);
        data_received = data_modulated + noise;
        soft_demodulated_output = 2*data_received;
        soft_demodulated_output(11521:end) = 0;
        [Y,actualnumiter,finalparitychecks] = ldpcDecode(soft_demodulated_output,cfgLDPCDec,maxnumiter);
        Res = repmat({zeros(cfgLDPCEnc.BlockLength,1)}, 1, height(P));
        Y_temp = soft_demodulated_output;
        L = Y_temp;
        for x = 1 : maxnumiter
            for y = 1:numClusters
                for z = 1 : BlockSize
                    % Initial State
                    idx1 = CN_neighbors{(y-1)*BlockSize+z};          % neighbor indices
                    vals = sum(L(idx1));                      % state value
                    % quantization
                    dc = length(idx1);
                    w = vals / dc   % normalize
                    % T = 2*sqrt(dc)/dc;         % Threshold for quantisation
                    T = 0.5;
                    if w < -T
                        q = 0;
                    elseif w < 0
                        q = 1;
                    elseif w < T
                        q = 2;
                    else
                        q = 3;
                    end
                    q
                    current_state(y,z) = q;
                    
                end
            end
           
            state_hard = current_state;
            for y = 1 : numClusters
                vec = state_hard(y,:);
                s(y) = 1 + sum(vec .* (4.^(length(vec)-1:-1:0)));
            end
            % Possible Actions in the current state
            vals1 = zeros(1, numClusters);

            for y = 1:numClusters
                vals1(i) = Q(s(y), y);
            end
            [~, a] = sort(vals1, 'descend');
            for y = 1 : length(a)
                [Y_out,res1] = ldpc_cluster(Y_temp,CN_neighbors,a(y),Res{a(y)},row_weight,BlockSize);
                Y_temp = Y_out;
                Res{a(y)} = res1;
            end
        end
        % Y_out = Y_temp;
        output_final_whole = Y_out < 0;
        output_final = output_final_whole(1:cfgLDPCEnc.NumInformationBits);
        ber(j) = biterr(output_final,bits);
        ber1(j) = biterr(Y,bits);
        j
        i
    end
    ber_t(i) = mean(ber)/cfgLDPCEnc.NumInformationBits;
    ber_t_1(i) = mean(ber1)/cfgLDPCEnc.NumInformationBits;
end
semilogy(SNR_db,ber_t_1)
semilogy(SNR,ber_t)
grid on