load("P_2176.mat");
P_new = P_2176 + 1;
P_sign = sign(P_new);
for i = 1 : height(P_sign)
    Sum{i} = sum(P_sign(i,:));
end
