function I = Jfun(sigma)
% J-function approximation for EXIT charts
% Input : sigma (scalar or vector, sigma >= 0)
% Output: I = J(sigma)

% Parameters
sigma_star = 1.6363;

aJ1 = -0.0421061;
bJ1 =  0.209252;
cJ1 = -0.00640081;

aJ2 =  0.00181491;
bJ2 = -0.142675;
cJ2 = -0.0822054;
dJ2 =  0.0549608;

% Preallocate
I = zeros(size(sigma));

% Regions
idx1 = (sigma >= 0) & (sigma <= sigma_star);
idx2 = (sigma > sigma_star) & (sigma < 10);
idx3 = (sigma >= 10);

% Polynomial region
I(idx1) = aJ1 .* sigma(idx1).^3 ...
        + bJ1 .* sigma(idx1).^2 ...
        + cJ1 .* sigma(idx1);

% Exponential region
I(idx2) = 1 - exp( aJ2 .* sigma(idx2).^3 ...
                 + bJ2 .* sigma(idx2).^2 ...
                 + cJ2 .* sigma(idx2) ...
                 + dJ2 );

% Saturation
I(idx3) = 1;

% Numerical safety
I = min(max(I,0),1);
end
