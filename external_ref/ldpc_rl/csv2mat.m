% p_mackey = zeros(48,96);
% 
% for i = 1 : 288
%     p_mackey(P(i,1)+1,P(i,2)+1) = 1;
% end
% save("p_mackey.mat","p_mackey")

snr_db = [0.5 1 1.5 2 2.5 3];

snr = 10.^(snr_db./10);
