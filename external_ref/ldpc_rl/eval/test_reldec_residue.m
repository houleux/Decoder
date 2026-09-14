load("wran_384_256.mat","wran_384_256");
BlockSize = 16;
H = sparse(logical(wran_384_256));
pcmatrix = H;
blocksize_wran = 16;
load("Q_wran_snr_0_new_reward.mat","Q")
[m, ~] = size(pcmatrix);
% P = zeros(m/BlockSize);
CN_neighbors = cell(m,1);
for c = 1:m
    CN_neighbors{c} = find(pcmatrix(c,:));
end
params.maxStateBits = 11;
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
% row_weight = sum((P+1) ~= 0, 2);
SNR_db = [0 0.5 1 1.5 2 2.5 3];

SNR = 10.^(SNR_db/10);
maxnumiter = 5;
for i =1 : 1
    current_state = zeros(m,10);
    for j = 1 : 20000
        bits = zeros(cfgLDPCEnc.NumInformationBits,1);
        codeword = ldpcEncode(bits,cfgLDPCEnc);
        codeword_1 = (codeword == 0);
        codeword_2 = (codeword == 1);
        data_modulated = sqrt(SNR(i)).*codeword_1 -sqrt(SNR(i)).*codeword_2;
        noise = randn(cfgLDPCEnc.BlockLength,1);
        data_received = data_modulated + noise;
        soft_demodulated_output = 2*data_received;
        [Y,actualnumiter,finalparitychecks] = ldpcDecode(soft_demodulated_output,cfgLDPCDec,maxnumiter);
        Res = repmat({zeros(cfgLDPCEnc.BlockLength,1)}, 1, height(H)/BlockSize);
        Y_temp = soft_demodulated_output;
        for k = 1 : m
            res{k} = zeros(1,numel(CN_neighbors{k}));
        end
        for x = 1:maxnumiter

            % ---------- INITIAL STATE ----------
            current_state = zeros(m, params.maxStateBits);
            res_quant = zeros(1,m);

            for v = 1:m
                idx1 = CN_neighbors{v};
                vals = Y_temp(idx1);
                vals = vals(:)';

                k = min(length(vals), params.maxStateBits);
                current_state(v,1:k) = vals(1:k);

                % ---------- QUANTIZATION ----------
                w = min(abs(vals(1:k) - res{v}(1:k)));

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

            % ---------- HARD STATE ----------
            state_hard_updated = current_state < 0;

            % ---------- PRECOMPUTE ----------
            pow2vec = 2.^(params.maxStateBits-1:-1:0);

            % ---------- TRACK USED CLUSTERS ----------
            used = false(1, height(pcmatrix)/BlockSize);

            % ---------- SEQUENTIAL SCHEDULING ----------
            for u = 1:height(height(pcmatrix)/BlockSize)

                % ===== STATE → INDEX =====
                s_new = 1 + state_hard_updated * pow2vec';

                % ===== Q LOOKUP (MULTI-TABLE) =====
                vals1 = zeros(1,m);
                for f = 1:m
                    vals1(f) = Q{res_quant(f)}(s_new(f), f);
                end

                % ===== CLUSTER AGGREGATION =====
                vals4 = zeros(1, height(height(pcmatrix)/BlockSize));
                for k = 1:height(height(pcmatrix)/BlockSize)
                    start_idx = (k-1)*BlockSize + 1;
                    end_idx   = k*BlockSize;
                    vals4(k) = sum(vals1(start_idx:end_idx));
                end

                % ===== MASK USED CLUSTERS =====
                temp = vals4;
                temp(used) = -inf;

                % ===== SELECT BEST =====
                [~, best_cluster] = max(temp);
                used(best_cluster) = true;

                % ===== BP UPDATE =====
                [Y_out, res1] = ldpc_cluster(...
                    Y_temp, CN_neighbors, best_cluster, ...
                    Res{best_cluster}, BlockSize);

                Y_temp = Y_out;
                Res{best_cluster} = res1;

                % ===== UPDATE STATE AFTER ACTION =====
                current_state = zeros(m, params.maxStateBits);
                res_quant = zeros(1,m);

                for v = 1:m
                    idx1 = CN_neighbors{v};
                    vals = Y_temp(idx1);
                    vals = vals(:)';

                    k = min(length(vals), params.maxStateBits);
                    current_state(v,1:k) = vals(1:k);

                    % re-quantize AFTER update
                    w = min(abs(vals(1:k) - res{v}(1:k)));

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

                state_hard_updated = current_state < 0;

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