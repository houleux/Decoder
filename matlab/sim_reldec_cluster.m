% Configuration
load("P_520_100.mat","P_520_100")
load("Q_1e5_P_520.mat","Q")
% Q = Q ./ max(abs(Q(:)));
P = P_520_100;
BlockSize = 10;
epsilon_test = 0.05;
pcmatrix = ldpcQuasiCyclicMatrix(BlockSize,P);
% Q = readmatrix('qtable_ep015000.csv');
% Q = Q(2:end, :);
% pcmatrix = sparse(logical(readmatrix('WRAN_irreg_384_256 (1).csv')));
%% 
[m, ~] = size(pcmatrix);
% P = zeros(m/BlockSize);
CN_neighbors = cell(m,1);
for c = 1:m
    CN_neighbors{c} = find(pcmatrix(c,:));
end
params.maxStateBits = 10;
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
SNR_db = -3;
% SNR_db = -2;
SNR = 10.^(SNR_db/10);
maxnumiter = 5;
for i =1 : length(SNR)
    current_state = zeros(m,10);
    for j = 1 : 100000
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
        for k = 1 : m
            res{k} = zeros(1,numel(CN_neighbors{k}));
        end
        for x = 1 : maxnumiter
            current_state = zeros(m, params.maxStateBits);
            for v = 1:m 
                idx1 = CN_neighbors{v}; % neighbor indices
                vals = Y_temp(idx1);
                % extract values % ensure row vector
                vals = vals(:)';
                k = min(length(vals), params.maxStateBits);
                current_state(v,1:k) = vals(1:k);
                % remaining elements already zero (no need to fill)
                w = min(abs(vals(1:k)-res{v}));
                if w < 0.25
                    q = 1;
                elseif w < 0.5
                    q = 2;
                elseif w < 0.75
                    q = 3;
                else
                    q = 4;
                end
                res_quant(v) = q;
            end
            % Hard Decoded State
            state_hard_updated = zeros(m,params.maxStateBits);
            for u = 1 : m
                state_hard_updated(u,:) = current_state(u, :) < 0;
            end
            for f = 1 : m 
                vec = state_hard_updated(f,:); 
                s_new(f) = 1 + sum(vec .* (2.^(length(vec)-1:-1:0)));
                vals1(f) = Q{res_quant(f)}(s_new(f),f);
            end
            % [~,a(x)] = max(vals1);
            % idx2 = CN_neighbors{a(x)};
            % vals2 = Y_temp(idx2)'- res{a(x)};
            % temp = tanh(vals2./2);
            % prodLq = prod(temp);
            % res{a(x)} = 2*atanh(prodLq ./ temp);
            % Y_temp(idx2) = vals2 + res{a(x)};
            vals4 = zeros(1, height(P));
            
            for k = 1:length(vals4)
                start_idx = (k-1)*BlockSize + 1;
                end_idx   = k*BlockSize;
                vals4(k) = sum(vals1(start_idx:end_idx));
            end
            [sortedVals, sortedIdx] = sort(vals4, 'descend');

            a(x,:) = sortedIdx;   % best action
            for u = 1 : height(P)
            % [~,a(x)] = max(vals4);
            [Y_out,res1] = ldpc_cluster(Y_temp,CN_neighbors,a(x,u),Res{a(x,u)},row_weight,BlockSize);
            Y_temp = Y_out;
            Res{a(x,u)} = res1;
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
