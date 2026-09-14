load('wran_384_256.mat','wran_384_256');
% load("P_520.mat","P_520")
% P = P_520;
H = sparse(logical(wran_384_256));
% H = ldpcQuasiCyclicMatrix(10,P);
cfgLDPCEnc = ldpcEncoderConfig(H);
cfgLDPCDec_l = ldpcDecoderConfig(H, "layered-bp");
cfgLDPCDec = ldpcDecoderConfig(H);
SNR_db = [0 1 2 3 4];
SNR = 10.^(SNR_db/10);

for i =1 : length(SNR)
    parfor j = 1 : 100000
        bits = randi([0 1], cfgLDPCEnc.NumInformationBits,1);
        codeword = ldpcEncode(bits,cfgLDPCEnc);
        codeword_1 = (codeword == 0);
        codeword_2 = (codeword == 1);
        %data_modulated = sqrt(SNR(i)).*real(pskmod(codeword, 2));
        data_modulated = sqrt(SNR(i)).*codeword_1 -sqrt(SNR(i)).*codeword_2;
        noise = randn(cfgLDPCEnc.BlockLength,1);
        data_received = data_modulated + noise;
        soft_demodulated_output = 2*data_received;
        [Y,actualnumiter,finalparitychecks] = ldpcDecode(soft_demodulated_output,cfgLDPCDec,5);
        [output_final,actualnumiter1,finalparitychecks1] = ldpcDecode(soft_demodulated_output,cfgLDPCDec_l,5);
        ber(j) = biterr(output_final,bits);
        ber1(j) = biterr(Y,bits);
        j
        i
    end
    ber_t(i) = mean(ber)/cfgLDPCEnc.NumInformationBits;
    ber_t_1(i) = mean(ber1)/cfgLDPCEnc.NumInformationBits;
end

figure;
semilogy(SNR, ber_t_1, 'b-o');      
hold on;
semilogy(SNR, ber_t, 'r-s');
hold off;
xlabel('SNR (dB)');
ylabel('BER');
legend('flood', 'layered');
grid on;


