d0 = '/home2/harshitlalwani/Decoder_100k/external_ref/ldpc_rl/checkpoints/sched';
names = {'wran_reldec_z1_ep500', 'wran_reldec_z4_ep500', 'wran_tanh_mi_z1_ep500', 'wran_tanh_mi_z4_ep500'};
for i = 1:numel(names)
    f = fullfile(d0, [names{i} '.mat']);
    d = load(f);  pol = d.pol;  meta = d.meta;
    save('-v7', f, 'pol', 'meta');
    printf('converted %s\n', names{i});
end
