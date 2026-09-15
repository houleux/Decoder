function sigma = Jinv(I)
% Inverse J-function approximation
% Input : I (scalar or vector, 0 <= I < 1)
% Output: sigma = J^{-1}(I)

% Parameters
I_star = 0.3646;

as1 = 1.09542;
bs1 = 0.214217;
cs1 = 2.33727;

as2 = 0.706692;
bs2 = 0.386013;
cs2 = -1.75017;

% Preallocate
sigma = zeros(size(I));

% Regions
idx1 = (I >= 0) & (I <= I_star);
idx2 = (I > I_star) & (I < 1);

% Polynomial + sqrt region
sigma(idx1) = as1 .* I(idx1).^2 ...
            + bs1 .* I(idx1) ...
            + cs1 .* sqrt(I(idx1));

% Logarithmic region
sigma(idx2) = -as2 .* log( bs2 .* (1 - I(idx2)) ) ...
              - cs2 .* I(idx2);

% Numerical safety
sigma = max(sigma,0);
end
