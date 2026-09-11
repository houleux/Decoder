function [LLR_final,Res] = ldpcdec_edge(i,j,Res_prev,LLR_in,pcmatrix,P)
blockSize = 32;
Res = Res_prev;
LLR_int = LLR_in - Res_prev;
p_temp = pcmatrix{i}.*LLR_int;
p_temp(p_temp == 0) = inf;
v = P(i,j): P(i,j) + blockSize-1;
v = rem(v,blockSize)+1;
for k = 1 : blockSize
    Res((j-1)*blockSize+v(k)) = 2*atanh((prod(tanh(p_temp(k,:)/2)))/tanh(p_temp(k,(j-1)*blockSize+v(k))/2));
end

LLR_final = (LLR_int + Res)';
Res = Res';
end




