load('P_2176.mat', 'P_2176')
P = P_2176;
R = 1-(height(P)/length(P));
% BlockSize = load('BlockSize_2176.mat');
Blocksize = 32;
for i = 1:height(P)
    pcmatrix{i} = full(ldpcQuasiCyclicMatrix(blockSize, P(i,:)));
end
pcmatrix1 = ldpcQuasiCyclicMatrix(Blocksize,P);
cfgLDPCEnc = ldpcEncoderConfig(pcmatrix1);
cfgLDPCDec_bp = ldpcDecoderConfig(pcmatrix1);
cfgLDPCDec = ldpcDecoderConfig(pcmatrix1,'layered-bp');
load('x_2176_2db.mat', 'ans')
x1 = ans;
load('y_2176_2db.mat', 'ans')
y1 = ans;
SNR_db = -1;
SNR = 10.^(SNR_db/10);
load('ldpc_precomp.mat')
for i =1 : length(SNR)
    parfor j = 1 : 1000
        bits = randi([0 1], cfgLDPCEnc.NumInformationBits,1);
        codeword = ldpcEncode(bits,cfgLDPCEnc);
        data_modulated = sqrt(SNR(i)) * (1 - 2*codeword);
        noise = randn(cfgLDPCEnc.BlockLength,1);
        data_received = data_modulated + noise;
        soft_demodulated_output = 2*data_received;
        [Y,actualnumiter,finalparitychecks] = ldpcDecode(soft_demodulated_output,cfgLDPCDec,5);
        Y_int2 = repmat({zeros(cfgLDPCEnc.BlockLength,1)}, height(P), 1);
        Y_temp2 = soft_demodulated_output;
        for k1 = 1 : length(y1)
            [Y_out2,Y_int2{x1(k1)}] = ldpcdec_edge(x1(k1),y1(k1),Y_int2{x1(k1)}',Y_temp2',pcmatrix,P);
            Y_temp2 = Y_out2;
        end
        output_final_whole = Y_out2 < 0;
        output_final = output_final_whole(1:cfgLDPCEnc.NumInformationBits);
        ber(j) = biterr(output_final,bits);
        ber1(j) = biterr(Y,bits);
        j
        i
    end
    ber_t(i) = mean(ber)/cfgLDPCEnc.NumInformationBits;
    ber_t_1(i) = mean(ber1)/cfgLDPCEnc.NumInformationBits;
end