function pol = sched_policy_pack(pol)
% SCHED_POLICY_PACK - make a policy safe to save: a 'map' store (cell of
% containers.Map) becomes plain arrays, pol.mapkeys{k} (cellstr of '0'/'1'
% state strings) and pol.mapvals{k} (Q values), which both MATLAB and Octave
% can write to and read from a .mat file. Inverse: sched_policy_unpack.m.
if isfield(pol, 'store') && strcmp(pol.store, 'map') && isfield(pol, 'maps')
    K = numel(pol.maps);
    pol.mapkeys = cell(1, K);  pol.mapvals = cell(1, K);
    for k = 1:K
        pol.mapkeys{k} = keys(pol.maps{k});
        v = values(pol.maps{k});
        pol.mapvals{k} = [v{:}];
    end
    pol = rmfield(pol, 'maps');
end
end
