% load('P_3840.mat')
load('P_2176.mat')
%load('P.mat')
P = P_3840;
blockSize = 384;
P1 = remove_one_row_submatrices(P);
pcmatrix = ldpcQuasiCyclicMatrix(blockSize,P);
for i = 1 : height(P)
    pcmatrix1{i} = ldpcQuasiCyclicMatrix(blockSize,P1{i});
end
for i = 1 : height(P)
    p_sub{i} = full(pcmatrix((i-1)*(blockSize)+1:i*blockSize,:));
end
cfgLDPCEnc = ldpcEncoderConfig(pcmatrix);
cfgLDPCDec_bp = ldpcDecoderConfig(pcmatrix);
cfgLDPCDec = ldpcDecoderConfig(pcmatrix,'layered-bp');
for i = 1 : height(P)
    cfgLDPCDec1{i} = ldpcDecoderConfig(pcmatrix1{i});
end
SNR_db = [1 1.1 1.2 1.3];
SNR = 10.^(SNR_db/10);

for i =1 : 1
    for j = 1 : 1
        bits = randi([0 1], cfgLDPCEnc.NumInformationBits,1);
        codeword = ldpcEncode(bits,cfgLDPCEnc);
        codeword_1 = (codeword == 0);
        codeword_2 = (codeword == 1);
        data_modulated = sqrt(SNR(i)).*codeword_1 -sqrt(SNR(i)).*codeword_2;
        noise = randn(cfgLDPCEnc.BlockLength,1);
        data_received = data_modulated + noise;
        soft_demodulated_output = 2*data_received;
        soft_demodulated_output(11521:end) = 0;
        [Y,actualnumiter,finalparitychecks] = ldpcDecode(soft_demodulated_output,cfgLDPCDec,5);
        Y_int = repmat({zeros(cfgLDPCEnc.BlockLength,1)}, 1, height(P));
        Y_int1 = repmat({zeros(cfgLDPCEnc.BlockLength,1)}, 1, height(P));
        Y_temp = soft_demodulated_output;
        Y_temp2 = soft_demodulated_output;
        for x = 1 : 1
            % x = scheduler_c_v(Y_temp,Y_int{1},Y_int{2},Y_int{3},Y_int{4},Y_int{5},Y_int{6});
            % Y_temp = Y_temp - Y_int{x};
            % Y_BP = ldpcDecode(Y_temp,cfgLDPCDec_bp,1,DecisionType='soft',OutputFormat='whole',MinSumScalingFactor=1,MinSumOffset=0);
            % Y_SUB = ldpcDecode(Y_temp,cfgLDPCDec1{x},1,DecisionType='soft',OutputFormat='whole',MinSumScalingFactor=1,MinSumOffset=0);
            % Y_int{x} = Y_BP - Y_SUB;
            % Y_out = Y_temp + Y_int{x};
            % Y_temp = Y_out;
            Y_temp = Y_temp - Y_int{x};
            Y_BP = ldpcDecode(Y_temp,cfgLDPCDec_bp,1,DecisionType='soft',OutputFormat='whole',MinSumScalingFactor=1,MinSumOffset=0);
            Y_SUB = ldpcDecode(Y_temp,cfgLDPCDec1{x},1,DecisionType='soft',OutputFormat='whole',MinSumScalingFactor=1,MinSumOffset=0);
            Y_int{x} = Y_BP - Y_SUB;
            Y_out = Y_temp + Y_int{x};
            Y_temp = Y_out;
        end
        [Y_out2,Y_int1{1}] = ldpcdec_cluster(1,Y_int1{1},Y_temp2,pcmatrix,P,blockSize) ;
        Y_temp2 = Y_out2;
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