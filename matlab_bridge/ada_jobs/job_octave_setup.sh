#!/bin/bash
#SBATCH -A research
#SBATCH -p u22
#SBATCH -c 4
#SBATCH --mem-per-cpu=2900M
#SBATCH --time=01:00:00
#SBATCH --job-name=oct_setup
#SBATCH --output=logs/%x-%j.out
set -uo pipefail
cd ~/Decoder_100k
module load u22/singularity-ce/4.2.2
mkdir -p ~/containers /scratch/$USER/sing_cache
export SINGULARITY_CACHEDIR=/scratch/$USER/sing_cache APPTAINER_CACHEDIR=/scratch/$USER/sing_cache
IMG=~/containers/octave_9.2.0.sif
[ -f $IMG ] || singularity pull $IMG docker://gnuoctave/octave:9.2.0
ls -la $IMG
OCT="singularity exec --bind $HOME $IMG octave-cli --no-gui --quiet"
$OCT --eval "disp(version)"
cd external_ref/ldpc_rl
$OCT --eval "addpath('lib'); rand('state',1); randn('state',1); printf('gradcheck %.2g\n', ppo_gradcheck()); t=tic; env=sched_env('wran',1); randn('state',5); s=sqrt(1/(2*env.code_rate*10^(2/10))); o=struct('i_max',5,'train',false); for f=1:200, sched_frame(2*(1+s*randn(1,env.n))/s^2, env, struct('kind','round_robin'), o); end; printf('wran round_robin %.1f ms/frame\n', toc(t)/200*1e3); env=sched_env('bg2_z10',1); s=sqrt(1/(2*env.code_rate*10^(2.5/10))); o.i_max=50; t=tic; for f=1:50, sched_frame(2*(1+s*randn(1,env.n))/s^2, env, struct('kind','round_robin'), o); end; printf('bg2 round_robin %.1f ms/frame\n', toc(t)/50*1e3);"
echo SETUP_DONE
