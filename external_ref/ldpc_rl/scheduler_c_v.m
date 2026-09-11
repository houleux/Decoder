function [x] = scheduler_c_v(Y_temp,Y_int1,Y_int2,Y_int3,Y_int4,Y_int5,Y_int6)
P = [
    16 17 22 24  9  3 14 -1  4  2  7 -1 26 -1  2 -1 21 -1  1  0 -1 -1 -1 -1
    25 12 12  3  3 26  6 21 -1 15 22 -1 15 -1  4 -1 -1 16 -1  0  0 -1 -1 -1
    25 18 26 16 22 23  9 -1  0 -1  4 -1  4 -1  8 23 11 -1 -1 -1  0  0 -1 -1
     9  7  0  1 17 -1 -1  7  3 -1  3 23 -1 16 -1 -1 21 -1  0 -1 -1  0  0 -1
    24  5 26  7  1 -1 -1 15 24 15 -1  8 -1 13 -1 13 -1 11 -1 -1 -1 -1  0  0
     2  2 19 14 24  1 15 19 -1 21 -1  2 -1 24 -1  3 -1  2  1 -1 -1 -1 -1  0
    ];
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
blockSize = 27;
pcmatrix = ldpcQuasiCyclicMatrix(blockSize,P);
pcmatrix1 = ldpcQuasiCyclicMatrix(blockSize,p1);
pcmatrix2 = ldpcQuasiCyclicMatrix(blockSize,p2);
pcmatrix3 = ldpcQuasiCyclicMatrix(blockSize,p3);
pcmatrix4 = ldpcQuasiCyclicMatrix(blockSize,p4);
pcmatrix5 = ldpcQuasiCyclicMatrix(blockSize,p5);
pcmatrix6 = ldpcQuasiCyclicMatrix(blockSize,p6);
cfgLDPCDec_bp = ldpcDecoderConfig(pcmatrix);
cfgLDPCDec1 = ldpcDecoderConfig(pcmatrix1);
cfgLDPCDec2 = ldpcDecoderConfig(pcmatrix2);
cfgLDPCDec3 = ldpcDecoderConfig(pcmatrix3);
cfgLDPCDec4 = ldpcDecoderConfig(pcmatrix4);
cfgLDPCDec5 = ldpcDecoderConfig(pcmatrix5);
cfgLDPCDec6 = ldpcDecoderConfig(pcmatrix6);

Y_temp1 = Y_temp - Y_int1;
Y_BP1 = ldpcDecode(Y_temp1,cfgLDPCDec_bp,1,DecisionType='soft',OutputFormat='whole',MinSumScalingFactor=1,MinSumOffset=0);
Y_SUB1 = ldpcDecode(Y_temp1,cfgLDPCDec1,1,DecisionType='soft',OutputFormat='whole',MinSumScalingFactor=1,MinSumOffset=0);
% R1 = Y_BP1 - Y_SUB1 - Y_int1;
R1 = Y_BP1 - Y_SUB1;

Y_temp2 = Y_temp - Y_int2;
Y_BP2 = ldpcDecode(Y_temp2,cfgLDPCDec_bp,1,DecisionType='soft',OutputFormat='whole',MinSumScalingFactor=1,MinSumOffset=0);
Y_SUB2 = ldpcDecode(Y_temp2,cfgLDPCDec2,1,DecisionType='soft',OutputFormat='whole',MinSumScalingFactor=1,MinSumOffset=0);
% R2 = Y_BP2 - Y_SUB2 - Y_int2;
R2 = Y_BP2 - Y_SUB2;

Y_temp3 = Y_temp - Y_int3;
Y_BP3 = ldpcDecode(Y_temp3,cfgLDPCDec_bp,1,DecisionType='soft',OutputFormat='whole',MinSumScalingFactor=1,MinSumOffset=0);
Y_SUB3 = ldpcDecode(Y_temp3,cfgLDPCDec3,1,DecisionType='soft',OutputFormat='whole',MinSumScalingFactor=1,MinSumOffset=0);
% R3 = Y_BP3 - Y_SUB3 - Y_int3;
R3 = Y_BP3 - Y_SUB3;

Y_temp4 = Y_temp - Y_int4;
Y_BP4 = ldpcDecode(Y_temp4,cfgLDPCDec_bp,1,DecisionType='soft',OutputFormat='whole',MinSumScalingFactor=1,MinSumOffset=0);
Y_SUB4 = ldpcDecode(Y_temp4,cfgLDPCDec4,1,DecisionType='soft',OutputFormat='whole',MinSumScalingFactor=1,MinSumOffset=0);
% R4 = Y_BP4 - Y_SUB4 - Y_int4;
R4 = Y_BP4 - Y_SUB4;

Y_temp5 = Y_temp - Y_int5;
Y_BP5 = ldpcDecode(Y_temp5,cfgLDPCDec_bp,1,DecisionType='soft',OutputFormat='whole',MinSumScalingFactor=1,MinSumOffset=0);
Y_SUB5 = ldpcDecode(Y_temp5,cfgLDPCDec5,1,DecisionType='soft',OutputFormat='whole',MinSumScalingFactor=1,MinSumOffset=0);
% R5 = Y_BP5 - Y_SUB5 - Y_int5;
R5 = Y_BP5 - Y_SUB5;

Y_temp6 = Y_temp - Y_int6;
Y_BP6 = ldpcDecode(Y_temp6,cfgLDPCDec_bp,1,DecisionType='soft',OutputFormat='whole',MinSumScalingFactor=1,MinSumOffset=0);
Y_SUB6 = ldpcDecode(Y_temp6,cfgLDPCDec6,1,DecisionType='soft',OutputFormat='whole',MinSumScalingFactor=1,MinSumOffset=0);
% R6 = Y_BP6 - Y_SUB6 - Y_int6;
R6 = Y_BP6 - Y_SUB6;

r1_max = max(abs(R1));
r2_max = max(abs(R2));
r3_max = max(abs(R3));
r4_max = max(abs(R4));
r5_max = max(abs(R5));
r6_max = max(abs(R6));

[~,x] = max([r1_max, r2_max, r3_max, r4_max, r5_max, r6_max]);
end
