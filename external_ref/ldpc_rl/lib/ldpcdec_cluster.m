function [LLR_final,Res] = ldpcdec_cluster(i,Res_prev,LLR_in,pcmatrix,P,BlockSize)
p_sub = pcmatrix((i-1)*(BlockSize)+1:i*BlockSize,:);
Res = Res_prev;
LLR_int = LLR_in - Res_prev;
p_temp = p_sub.*LLR_int';
p_temp(p_temp == 0) = inf;
for i = 1 : height(p_sub)
    for j = 1 : length(p_sub)
        if p_temp(i,j) == inf
            Res(j) = Res(j);
        else
            Res(j) = 2*atanh((prod(tanh(p_temp(i,:)/2)))/tanh(p_temp(i,j)/2));
        end
    end
end
LLR_final = (LLR_int + Res);
end