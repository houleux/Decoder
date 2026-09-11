function [LLR_out,Res_out,cluster] = ldpc_residue_cluster(LLR_in,C,Res,row_weight,BlockSize)
for i = 1 : numel(Res)
    LLR_int{i} = LLR_in - Res{i};
    Res_int{i} = zeros(length(LLR_in),1);
end

for i = 1 : numel(C)
    for j = 1 : BlockSize
        idx = C{i}(j,1:row_weight(i)); 
        llr_temp = LLR_int{i}(idx);
        temp = tanh(llr_temp./2);
        prodLq = prod(temp);  
        Res_int{i}(idx) = 2*atanh(prodLq ./ temp);
    end
end
for i = 1 : numel(C)
    u(i) = max(abs(LLR_int{i}+Res_int{i}-LLR_in));  % mean(abs(Res{i}))
end
[~,cluster] = max(u);
Res_out = Res_int{cluster};
LLR_out = LLR_int{cluster} + Res_int{cluster};
end