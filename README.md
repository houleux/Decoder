# RL for LDPC Decoders

This repository contains the infrastructure for training, evaluating, and visualizing Reinforcement Learning (RL) agents for LDPC Decoding. The pipeline provides a unified, highly parallelized evaluation engine backed by a **central PostgreSQL database** (hosted on Neon, shared by every machine) and a dynamic **Flask Web UI** for live visualization.

---

## 0. Setup (any machine)

```bash
git clone https://github.com/houleux/Decoder.git && cd Decoder
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Build the vendored ldpc decoder (Cython) in place. run_experiments.py imports it from ldpc/src_python.
(cd ldpc && python setup.py build_ext --inplace)

# Database credentials: copy the template and paste the Neon *pooled* connection string.
cp .env.example .env    # then edit: EXPDB_URL=postgresql://...
python3 cli/exp.py ls   # should list existing configs
```

`.env` is gitignored; never commit it. Every script reads `EXPDB_URL` from the environment or from `.env`, and fails immediately if it is missing (there is no local fallback database).

The database is reached over **HTTPS** (Neon's SQL-over-HTTP endpoint on port 443), not the usual Postgres port 5432, because the Ada cluster's firewall only allows outbound web ports. Nothing extra needs installing for this.

Creating a brand-new, empty database (only needed once, ever): `python3 cli/exp.py init-db`.

---

## 1. Unified Experiment Runner

All experiments are orchestrated through a single unified script: `run_experiments.py`. This script automatically handles:
1. **Training Phase**: Trains the specified RL agents for a set number of episodes and saves the checkpoints. It automatically skips training if a checkpoint already exists.
2. **Evaluation Phase**: Leverages Python multiprocessing (`forkserver`) to evaluate the agents across multiple Signal-to-Noise Ratios (SNRs). 
3. **Database Logging**: All evaluation progress is chunked (e.g., every 100 frames) and committed to the central database, so results from every machine land in one place.

### Usage

```bash
python3 run_experiments.py \
    --matrix matrices/H_AB_LDPC_500.csv \
    --methods flooding reldec ave_tanh_ave_mi ave_mi_ave_mi \
    --z-vals 1 \
    --train-snrs 1.0 1.5 2.0 2.5 3.0 \
    --eval-snrs 1.0 1.5 2.0 2.5 3.0 \
    --train-episodes 100 \
    --max-frames 1000 \
    --workers 40
```

> [!TIP]
> The evaluation loop is entirely interruptible! If the script crashes or you kill it, running it again will automatically pick up right where it left off, down to the exact SNR and chunk, because progress is stored in the database.

> [!WARNING]
> Don't run the *same* sweep on two machines at once: both start from the same `frames_done`, simulate identical frames (the RNG seed is `seed + frames_done`), and both commit them. Split work across machines by method, SNR, or config instead.

---

## 2. Live Web Dashboard (Plotting)

Instead of generating hundreds of redundant `.png` files, all plotting and filtering is done dynamically in the browser. The dashboard fetches the latest data from the central database and renders Matplotlib charts in memory. Because the database is central, you can run the dashboard on any machine with a `.env`, including your laptop.

### Running the Dashboard
Start the Flask backend:
```bash
python3 webui/app.py
```
Then navigate to `http://localhost:5000` in your web browser.

### Features
- **Dynamic Filters**: Automatically extracts and lets you filter by all recorded metadata (Matrix, Agent Method, Cluster Size `Z`, etc.).
- **Live Updating**: As `run_experiments.py` runs in the background, simply hit **"Plot Selected"** on the dashboard to redraw the curves with the most up-to-date intermediate results.
- **Save on Demand**: Found a plot you like? Click the "Download Plot" button to save it locally.

---

## 3. Working on a SLURM Cluster

The infrastructure is explicitly designed to run seamlessly on remote HPC environments like SLURM.

### Database
Jobs on compute nodes write straight to the central database over HTTPS. `sbatch` jobs inherit `EXPDB_URL` from the submitting shell, or read it from the repo's `.env`. **You do not need to configure or request a SQL server from your cluster admins.**

### Viewing the UI via SSH
To view the Web UI on your personal laptop while it runs on the cluster, use **SSH local port forwarding**:

1. SSH into the login node:
   ```bash
   ssh -L 5000:localhost:5000 your_username@cluster_address
   ```
2. Start the UI:
   ```bash
   python3 webui/app.py
   ```
3. Open `http://localhost:5000` on your laptop.

> [!NOTE]
> The Web UI uses headless Matplotlib (`Agg` backend). It will safely generate plots on headless nodes without crashing or requiring X11 forwarding.

### Compute Nodes & Multiprocessing
When launching `run_experiments.py` via an `sbatch` script or `srun`, ensure you explicitly request enough CPU cores to match your `--workers` count (e.g., `#SBATCH --cpus-per-task=40`). If you don't, SLURM's cgroups may throttle all 40 Python processes onto a single CPU core!
