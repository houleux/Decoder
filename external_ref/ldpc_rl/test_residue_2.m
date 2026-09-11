% Configuration
load('P_3840.mat')
load('P_2176.mat')
% load('P.mat')
P = P_2176;
BlockSize = 32;
pcmatrix = ldpcQuasiCyclicMatrix(BlockSize,P);
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
temp = 0;
for i = 1 : height(P)
    for j = 1 : BlockSize
        for k = 1 : row_weight(i)
            C{i}(j,k) = columnIndex(temp + (row_weight(i)*(j-1)) + k);
        end
    end
    temp = temp + (row_weight(i)*(j-1))+k;
    temp1(i) = temp;
end

SNR_db = [-3 -2 -1 0 1 2 3 4];
SNR = 10.^(SNR_db/10);
maxnumiter = 2;
for i =1 : length(SNR)
    parfor j = 1 : 1000
        bits = randi([0 1], cfgLDPCEnc.NumInformationBits,1);
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
        for x = 1 : maxnumiter*height(P)
            [Y_out,res1,t] = ldpc_residue_cluster(Y_temp,C,Res,row_weight,BlockSize);
            Y_temp = Y_out;
            Res{t} = res1;
        end
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