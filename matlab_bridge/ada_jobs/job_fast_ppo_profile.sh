#!/bin/bash
#SBATCH -A research
#SBATCH -p u22
#SBATCH -c 12
#SBATCH --mem-per-cpu=2900M
#SBATCH --time=02:00:00
#SBATCH --job-name=ml_fast_ppo_profile
#SBATCH --output=logs/%x-%j.out
# MATLAB smoke test + ppo vs fast_ppo training-speed profile (WRAN z=4,16).
set -uo pipefail
cd ~/Decoder_100k/external_ref/ldpc_rl
hostname; date
$HOME/matlab_R2024a/bin/matlab -licmode onlinelicensing -batch "run('$HOME/Decoder_100k/matlab_bridge/ada_jobs/fast_ppo_profile.m')"
echo JOB_PROFILE_DONE rc=$?; date
