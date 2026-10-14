function pol = sched_policy_unpack(pol)
% SCHED_POLICY_UNPACK - rebuild the containers.Map store of a policy saved by
% sched_policy_pack.m. Policies with other stores are returned unchanged.
if isfield(pol, 'store') && strcmp(pol.store, 'map') && ~isfield(pol, 'maps')
    K = numel(pol.mapkeys);
    pol.maps = cell(1, K);
    for k = 1:K
        M = containers.Map('KeyType', 'char', 'ValueType', 'double');
        ks = pol.mapkeys{k};  vs = pol.mapvals{k};
        if ischar(ks), ks = cellstr(ks); end
        for i = 1:numel(ks), M(ks{i}) = vs(i); end
        pol.maps{k} = M;
    end
    pol = rmfield(pol, {'mapkeys', 'mapvals'});
end
end
