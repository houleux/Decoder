function [LLR_out1,Res_int] = ldpc_cluster(LLR_in,C,a,Res,row_weight,BlockSize)

Res_int = zeros(length(LLR_in),1);
LLR_int = LLR_in;
LLR_int = LLR_int - Res;
for j = 1 : BlockSize
    idx = C{a}(j,1:row_weight(a));
    llr_temp = LLR_int(idx);
    temp = tanh(llr_temp./2);
    prodLq = prod(temp);
    Res_int(idx) = 2*atanh(prodLq ./ temp);
end
LLR_out1 = LLR_int + Res_int;
end