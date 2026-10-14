function mi = mi_llr(llr)
% MI_LLR - mutual information of an LLR via the J-function approximation.
% Element-wise; identical constants to this repo's rl/rewards/mi_utils.py.
s = sqrt(max(2*llr, 0));
mi = ones(size(s));
a = s <= 1.6363;
mi(a) = -0.0421061*s(a).^3 + 0.209252*s(a).^2 - 0.00640081*s(a);
b = (s > 1.6363) & (s < 10);
mi(b) = 1 - exp(0.00181491*s(b).^3 - 0.142675*s(b).^2 - 0.0822054*s(b) + 0.0549608);
mi = min(max(mi, 0), 1);
end
