load("P.mat");
p1 = [
    25 12 12  3  3 26  6 21 -1 15 22 -1 15 -1  4 -1 -1 16 -1  0  0 -1 -1 -1
    25 18 26 16 22 23  9 -1  0 -1  4 -1  4 -1  8 23 11 -1 -1 -1  0  0 -1 -1
     9  7  0  1 17 -1 -1  7  3 -1  3 23 -1 16 -1 -1 21 -1  0 -1 -1  0  0 -1
    24  5 26  7  1 -1 -1 15 24 15 -1  8 -1 13 -1 13 -1 11 -1 -1 -1 -1  0  0
     2  2 19 14 24  1 15 19 -1 21 -1  2 -1 24 -1  3 -1  2  1 -1 -1 -1 -1  0
    ];
p2 = [
    16 17 22 24  9  3 14 -1  4  2  7 -1 26 -1  2 -1 21 -1  1  0 -1 -1 -1 -1
    25 18 26 16 22 23  9 -1  0 -1  4 -1  4 -1  8 23 11 -1 -1 -1  0  0 -1 -1
     9  7  0  1 17 -1 -1  7  3 -1  3 23 -1 16 -1 -1 21 -1  0 -1 -1  0  0 -1
    24  5 26  7  1 -1 -1 15 24 15 -1  8 -1 13 -1 13 -1 11 -1 -1 -1 -1  0  0
     2  2 19 14 24  1 15 19 -1 21 -1  2 -1 24 -1  3 -1  2  1 -1 -1 -1 -1  0
    ];
p3 = [
    16 17 22 24  9  3 14 -1  4  2  7 -1 26 -1  2 -1 21 -1  1  0 -1 -1 -1 -1
    25 12 12  3  3 26  6 21 -1 15 22 -1 15 -1  4 -1 -1 16 -1  0  0 -1 -1 -1
     9  7  0  1 17 -1 -1  7  3 -1  3 23 -1 16 -1 -1 21 -1  0 -1 -1  0  0 -1
    24  5 26  7  1 -1 -1 15 24 15 -1  8 -1 13 -1 13 -1 11 -1 -1 -1 -1  0  0
     2  2 19 14 24  1 15 19 -1 21 -1  2 -1 24 -1  3 -1  2  1 -1 -1 -1 -1  0
    ];
p4 = [
    16 17 22 24  9  3 14 -1  4  2  7 -1 26 -1  2 -1 21 -1  1  0 -1 -1 -1 -1
    25 12 12  3  3 26  6 21 -1 15 22 -1 15 -1  4 -1 -1 16 -1  0  0 -1 -1 -1
    25 18 26 16 22 23  9 -1  0 -1  4 -1  4 -1  8 23 11 -1 -1 -1  0  0 -1 -1
    24  5 26  7  1 -1 -1 15 24 15 -1  8 -1 13 -1 13 -1 11 -1 -1 -1 -1  0  0
     2  2 19 14 24  1 15 19 -1 21 -1  2 -1 24 -1  3 -1  2  1 -1 -1 -1 -1  0
    ];
p5 = [
    16 17 22 24  9  3 14 -1  4  2  7 -1 26 -1  2 -1 21 -1  1  0 -1 -1 -1 -1
    25 12 12  3  3 26  6 21 -1 15 22 -1 15 -1  4 -1 -1 16 -1  0  0 -1 -1 -1
    25 18 26 16 22 23  9 -1  0 -1  4 -1  4 -1  8 23 11 -1 -1 -1  0  0 -1 -1
     9  7  0  1 17 -1 -1  7  3 -1  3 23 -1 16 -1 -1 21 -1  0 -1 -1  0  0 -1
     2  2 19 14 24  1 15 19 -1 21 -1  2 -1 24 -1  3 -1  2  1 -1 -1 -1 -1  0
    ];
p6 = [
    16 17 22 24  9  3 14 -1  4  2  7 -1 26 -1  2 -1 21 -1  1  0 -1 -1 -1 -1
    25 12 12  3  3 26  6 21 -1 15 22 -1 15 -1  4 -1 -1 16 -1  0  0 -1 -1 -1
    25 18 26 16 22 23  9 -1  0 -1  4 -1  4 -1  8 23 11 -1 -1 -1  0  0 -1 -1
     9  7  0  1 17 -1 -1  7  3 -1  3 23 -1 16 -1 -1 21 -1  0 -1 -1  0  0 -1
    24  5 26  7  1 -1 -1 15 24 15 -1  8 -1 13 -1 13 -1 11 -1 -1 -1 -1  0  0
    ];
% load("p2.mat");
% load("p3.mat");
% load("p4.mat");
% load("p5.mat");
% load("p6.mat");
blockSize = 27;
pcmatrix = ldpcQuasiCyclicMatrix(blockSize,P);
pcmatrix1 = ldpcQuasiCyclicMatrix(blockSize,p1);
pcmatrix2 = ldpcQuasiCyclicMatrix(blockSize,p2);
pcmatrix3 = ldpcQuasiCyclicMatrix(blockSize,p3);
pcmatrix4 = ldpcQuasiCyclicMatrix(blockSize,p4);
pcmatrix5 = ldpcQuasiCyclicMatrix(blockSize,p5);
pcmatrix6 = ldpcQuasiCyclicMatrix(blockSize,p6);
cfgLDPCEnc = ldpcEncoderConfig(pcmatrix);
cfgLDPCDec_bp = ldpcDecoderConfig(pcmatrix);
cfgLDPCDec = ldpcDecoderConfig(pcmatrix,'layered-bp');
cfgLDPCDec1{1} = ldpcDecoderConfig(pcmatrix1);
cfgLDPCDec1{2} = ldpcDecoderConfig(pcmatrix2);
cfgLDPCDec1{3} = ldpcDecoderConfig(pcmatrix3);
cfgLDPCDec1{4} = ldpcDecoderConfig(pcmatrix4);
cfgLDPCDec1{5} = ldpcDecoderConfig(pcmatrix5);
cfgLDPCDec1{6} = ldpcDecoderConfig(pcmatrix6);
SNR_db = 3;
SNR = 10.^(SNR_db/10);
% load('x_3db.mat');
% x1 = ans;
% load('y_3db.mat','x');
y1 = [1     2     3     4     5     6     7  9    10    11    13    15    17    19    20];
x1 = ones(1, length(y1));
c = [4	5	6	2	1	3	4	5	6	2	1	3	4	5	6	2	1	3	4	5	6	2	1	3	4	5	6	2	1	3];
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
        [Y,actualnumiter,finalparitychecks] = ldpcDecode(soft_demodulated_output,cfgLDPCDec,5);
        Y_int = cell(6,1);
        for t = 1:6
            Y_int{t} = zeros(cfgLDPCEnc.BlockLength,1);
        end
        Y_temp = soft_demodulated_output;
        for k = 1 : 30
            x = c(k);
            Y_temp = Y_temp - Y_int{x};
            Y_BP = ldpcDecode(Y_temp,cfgLDPCDec_bp,1,DecisionType='soft',OutputFormat='whole',MinSumScalingFactor=1,MinSumOffset=0);
            Y_SUB = ldpcDecode(Y_temp,cfgLDPCDec1{x},1,DecisionType='soft',OutputFormat='whole',MinSumScalingFactor=1,MinSumOffset=0);
            Y_int{x} = Y_BP - Y_SUB;
            Y_out = Y_temp + Y_int{x};
            Y_temp = Y_out;
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