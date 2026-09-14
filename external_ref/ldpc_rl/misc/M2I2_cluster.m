load('P_2176.mat', 'P_2176')
P = P_2176;
% load('P')
% BlockSize = load('BlockSize_2176.mat');
Blocksize = 32;
R = 1-(height(P)/length(P));
P_new = P + 1;
P_sign = sign(P_new);
SNR_db = 2;
iterations = 5;
snr = 10.^(SNR_db/10);
I_C2V_in = zeros(height(P),length(P));
I_V_in = J_approx(sqrt(8*R*snr))*ones(1,length(P)*Blocksize);
I_V2C= J_approx(sqrt(8*R*snr))*ones(height(P),length(P)).*P_sign;
row = zeros(1,nnz(P_new)*iterations);
col = zeros(1,nnz(P_new)*iterations);
val = zeros(1,nnz(P_new)*iterations);
c = zeros(1,iterations*height(P));
for k = 1 : iterations*height(P)    %nnz(P_new)*iterations
    I_C2V = zeros(height(P),length(P));
    for j = 1 : length(P)
        for i = 1 : height(P)
            if P_sign(i,j) == 0
                I_C2V(i,j) = 0;
            else
                I_C2V(i,j) = 1 - J_approx(sqrt(sum(Jinv_approx((1-I_V2C(i,:)).*P_sign(i,:)).^2)-(Jinv_approx(1-I_V2C(i,j))^2)));
            end
        end
    end
    r = I_C2V - I_C2V_in;
    r_c = sum(r,2);
    [val(k), idx] = max(r_c(:));
    c(k) = idx;
    P_sign_sub = P_sign(c(k),:);
    idx_c = find(P_sign_sub);
    I_C2V_in(c(k),:) = I_C2V(c(k),:); 
    for j = 1 : length(idx_c)
    for i = 1 : height(P)
        if P_sign(i,idx_c(j)) == 0
            I_V2C(i,idx_c(j)) = 0;
        else
            I_V2C(i,idx_c(j)) = J_approx(sqrt(sum(Jinv_approx((I_C2V_in(:,idx_c(j))).*P_sign(:,idx_c(j))).^2)-(Jinv_approx(I_C2V_in(i,idx_c(j)))^2) + 8*R*snr));
        end
    end
    end
end