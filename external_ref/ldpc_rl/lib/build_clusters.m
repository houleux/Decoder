function clusters = build_clusters(m, z)
% BUILD_CLUSTERS - contiguous groups of z check-node indices (1-indexed).
%
% Matches this repo's Python clustering convention exactly --
% rl/agents/reldec.py::ReldecAgent.__init__:
%     cn_indices = np.arange(self.m)
%     self.clusters = [cn_indices[i:i+z] for i in range(0, self.m, z)]
% (0-indexed there; this is the 1-indexed MATLAB equivalent.) z=1 gives one
% cluster per check node -- the same granularity Python's RoundRobinDecoder
% uses, just expressed as "clusters of size 1" here for a uniform interface
% across z values. If z does not evenly divide m, the LAST cluster is
% smaller than z (same behaviour as Python's slicing, which silently
% truncates the final slice rather than padding it).
%
% INPUTS
%   m - number of check nodes
%   z - cluster size
% OUTPUT
%   clusters - 1 x numClusters cell array; clusters{k} is a row vector of
%              1-indexed CN indices belonging to cluster k, in increasing order

if z < 1
    error('build_clusters:badZ', 'z must be >= 1, got %d', z);
end

clusters = {};
i = 1;
while i <= m
    j = min(i + z - 1, m);
    clusters{end+1} = i:j; %#ok<AGROW>
    i = j + 1;
end
end
