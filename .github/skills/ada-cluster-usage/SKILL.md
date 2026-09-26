---
name: ada-cluster-usage
description: 'Use when working on IIIT Ada HPC: SSH access, SLURM interactive or batch jobs, research account job submission, GPU requests, job monitoring, pending reason triage, storage hygiene, and Jupyter/VS Code tunneling.'
argument-hint: 'Task on Ada (interactive debug, sbatch run, monitor, or Jupyter tunnel)'
---

# Ada Cluster Usage

Use this skill to run ML or decoding experiments on the IIIT Ada cluster with safe defaults and repeatable SLURM workflows.

## When to Use
- Need to run training or evaluation on Ada compute nodes.
- Need to choose between `srun` interactive debugging and `sbatch` batch execution.
- Need robust monitoring (`squeue`, `sacct`, `scontrol`) and pending-job diagnosis.
- Need to launch JupyterLab or TensorBoard from compute nodes with SSH tunnels.

## Identity and Defaults
- Identity reference: `harshit.lalwani@research.iiit.ac.in`
- Default account: `research`
- Default partition for GPU work: `long`
- Recommended explicit QoS for research account: `medium`

If your actual Ada login differs from your email identifier, always use the Ada login username in commands. The Ada login username is `harshitlalwani`.

Authentication on Ada supports SSH keys. The laptop `LAPTOP-5AEL1NPO` (both its WSL `root` account and its Windows `harsh` account) is already authorized, so from there `ssh ada` / `ssh gnodeXX` need no password once the SSH config in section 8 is in place. From any other machine, or if a command prompts for a password, follow section 8 to set up a key rather than asking the user to type a password.

## Workflow

### 1. Pre-flight checks
1. Ensure VPN is active when off-campus.
2. SSH to head node:
   - `ssh -X <ada_username>@ada.iiit.ac.in`
3. Confirm account/QoS associations:
   - `sacctmgr show assoc user=$USER format=Account,QOS,DefaultQOS`
4. Confirm available partitions/nodes:
   - `sinfo -a`

Completion checks:
- SSH works without repeated auth failures.
- `research` account appears in association output.

### 2. Decide execution mode
- Use **interactive** (`srun --pty`) for debugging, environment setup, and short validation.
- Use **batch** (`sbatch`) for long training/evaluation runs and reproducibility.

Decision rule:
- If runtime is uncertain or you need live shell iteration, start with interactive.
- If runtime is long/repeatable, submit a batch job.

### 3. Interactive job template (debug path)
Use one GPU with balanced CPU/memory:

```bash
srun --pty \
  --partition=long \
  -A research \
  --qos=medium \
  --gres=gpu:1 \
  -c 10 \
  --mem-per-cpu=2G \
  bash -l
```

Inside the allocation:
1. Activate environment/modules.
2. Run a small sanity test first.
3. If successful, move to `sbatch` for long runs.

Completion checks:
- Prompt changes to compute node host (`gnodeXX`).
- GPU visible with `nvidia-smi`.

### 4. Batch job template (production path)
Create a script like `job_train.sh`:

```bash
#!/bin/bash
#SBATCH -A research
#SBATCH --qos=medium
#SBATCH --partition=long
#SBATCH -n 10
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=2G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

# Optional module setup
# module load u18/cuda/11.6

cd "$SLURM_SUBMIT_DIR"
source .venv/bin/activate
python RELDEC/train_reldec.py --help
```

Submit and monitor:
- `mkdir -p logs`
- `sbatch job_train.sh`
- `squeue -u $USER`
- `tail -f logs/<jobname>_<jobid>.out`

Completion checks:
- Job enters `R` (running) or expected `PD` (pending with valid reason).
- Log file is created and updates during runtime.

### 5. Pending job triage
Use:
- `squeue -u $USER -o "%.18i %.9P %.20j %.8u %.2t %.10M %.6D %R"`
- `scontrol show job <job_id>`

Common reasons and actions:
- `Priority`: wait or reduce resources.
- `Resources`: request fewer GPUs/CPUs or different partition.
- `QOSMaxJobsLimit` / `QOSMaxCpuPerUserLimit`: reduce concurrent jobs or adjust QoS/account.
- `AssocGrpGRES` / `AssocGrpCpuLimit`: account-level quota reached; wait or coordinate with admins.
- `ReqNodeNotAvail`: remove strict node pinning or pick another node.

### 6. Storage policy and data movement
Use storage intentionally:
- `/home/$USER`: code, envs, small artifacts (backed up, quota-limited).
- `/share1` (and `/share3` for eligible users): longer-term data.
- `/scratch` and `/ssd_scratch`: temporary high-volume runtime data; purge windows apply (typically about 7-10 days).

Transfer data:
- Local to Ada: `rsync -avz <local_dir> <ada_username>@ada.iiit.ac.in:`
- Ada to local: `rsync -avz <ada_username>@ada.iiit.ac.in:<remote_dir> .`

Completion checks:
- Critical results copied out of scratch before cleanup windows.

### 7. JupyterLab/TensorBoard from compute node
1. On Ada, start JupyterLab on a compute node inside your allocation:
   - `jupyter lab --no-browser --port=<port2>`
2. On local machine, tunnel via head node:
   - `ssh -L <port1>:localhost:<port2> -J <ada_username>@ada.iiit.ac.in <ada_username>@<gnodeXX>`
3. Open `http://localhost:<port1>` locally.

Completion checks:
- Browser opens Jupyter token/password page.
- Kernel runs on compute node, not head node.

### 8. Passwordless SSH (keys) and SSH config
Needed for Claude Code or any other non-interactive tool, which cannot type a password or passphrase. `$HOME` is shared across Ada, so one entry in `~/.ssh/authorized_keys` works for the head node and every compute node.

On the local machine (inside WSL if the tool runs in WSL; WSL and Windows have separate `~/.ssh` directories):

1. Check for an existing key: `cat ~/.ssh/id_ed25519.pub`. If there is none, create one with `ssh-keygen -t ed25519` and leave the passphrase empty, or keep a passphrase and load the key into `ssh-agent` with `ssh-add` every session.
2. Authorize it on Ada (asks for the password one last time):
   - Linux/macOS/WSL: `ssh-copy-id harshitlalwani@ada.iiit.ac.in`
   - Windows PowerShell: `type $env:USERPROFILE\.ssh\id_ed25519.pub | ssh harshitlalwani@ada.iiit.ac.in "cat >> ~/.ssh/authorized_keys"`
   - Or, from a session already on Ada, append the `.pub` line to `~/.ssh/authorized_keys` (the directory must be `700` and the file `600`).
3. Add `~/.ssh/config`, then `chmod 600 ~/.ssh/config`:

```sshconfig
Host ada
    HostName ada.iiit.ac.in
    User harshitlalwani
    IdentityFile ~/.ssh/id_ed25519
    ServerAliveInterval 60

Host gnode*
    User harshitlalwani
    ProxyJump ada
    IdentityFile ~/.ssh/id_ed25519
    StrictHostKeyChecking accept-new
```

4. Accept host keys once interactively (`ssh ada hostname`, then `ssh gnodeXX hostname`), then verify non-interactively:
   - `ssh -o BatchMode=yes ada hostname` should print `ada.iiit.ac.in`
   - `ssh -o BatchMode=yes gnodeXX hostname` should print `gnodeXX`

Gotchas:
- `root@ada's password:` means SSH sent the local username (`root` in WSL) because the config's `User` line is missing or not being read. Ada has no root login, so no password will work.
- `Host key verification failed` under `BatchMode=yes` means the host key has never been accepted, since BatchMode suppresses the prompt. Connect once without BatchMode. If you get `REMOTE HOST IDENTIFICATION HAS CHANGED`, run `ssh-keygen -R <host>` first.
- `Could not resolve hostname gnodeXX` means the `Host gnode*` / `ProxyJump` block is missing. Compute node names resolve only from inside Ada.
- Compute nodes use `pam_slurm_adopt`, so SSH to a `gnodeXX` is allowed only while you have a running job on that node. Otherwise connect to `ada`.
- The `connection is not using a post-quantum key exchange algorithm` warning is harmless (Ada's server is older than the client).

For VS Code Remote SSH, connect to `ada` or `gnodeXX` using the same config.

## Quality Criteria
- Never run long GPU jobs on login/head node.
- Every long run has an `sbatch` script and log files.
- Requested resources are justified and not over-provisioned.
- Output and checkpoints are copied from scratch to persistent storage.
- Job status and pending reasons are inspected before re-submitting repeatedly.

## Quick Prompts
- "Use ada-cluster-usage to prepare an sbatch script for RELDEC training on 1 GPU."
- "Use ada-cluster-usage to debug why my SLURM job is pending."
- "Use ada-cluster-usage to set up Jupyter tunneling from gnode47."
