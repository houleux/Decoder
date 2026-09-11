function [x] = ldpc_cluster_residue(LLR_in,Res_prev,P,p_sub)
for i =1 : height(P)
    Res{i} = Res_prev{i};
    LLR_int{i} = LLR_in - Res_prev{i};
    p_temp{i} = p_sub.*LLR_int';
    p_temp{i}(p_temp{i} == 0) = inf;
end
for k = 1 : height(P)
    for i = 1 : height(p_sub)
        for j = 1 : length(p_sub)
            if p_temp(i,j) == inf
                Res{k}(j) = Res{k}(j);
            else
                Res{k}(j) = 2*atanh((prod(tanh(p_temp(i,:)/2)))/tanh(p_temp(i,j)/2));
            end
        end
    end
end
for i = 1 : length(P)
    u(i) = max(abs(Res{i}));  % mean(abs(Res{i}))
end
[~,x] = max(u);
end