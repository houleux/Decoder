function net = ppo_init(D, hidden)
% PPO_INIT - weights of the PPO actor-critic (see ppo_lossgrad.m for the
% architecture). D = per-cluster features + 3 global features.
% Initialisation is PyTorch's nn.Linear default: weights and biases uniform in
% +-1/sqrt(fan_in). Uses the global random stream. Biases are row vectors.
u = @(r, c, fan) (2*rand(r, c) - 1) / sqrt(fan);
net.W1 = u(hidden, D, D);            net.b1 = u(1, hidden, D);
net.W2 = u(hidden, hidden, hidden);  net.b2 = u(1, hidden, hidden);
net.W3 = u(1, hidden, hidden);       net.b3 = u(1, 1, hidden);
net.E1 = u(hidden, D, D);            net.c1 = u(1, hidden, D);
net.E2 = u(hidden, hidden, hidden);  net.c2 = u(1, hidden, hidden);
net.C1 = u(hidden, hidden + 3, hidden + 3);  net.d1 = u(1, hidden, hidden + 3);
net.C2 = u(1, hidden, hidden);       net.d2 = u(1, 1, hidden);
end
