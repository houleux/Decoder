# RL for LDPC Decoders

This repository contains the infrastructure for training, evaluating, and visualizing Reinforcement Learning (RL) agents for LDPC Decoding. The pipeline has been overhauled to provide a unified, highly parallelized evaluation engine backed by **DuckDB** and a dynamic **Flask Web UI** for live visualization.

---

## 1. Unified Experiment Runner

All experiments are orchestrated through a single unified script: `run_experiments.py`. This script automatically handles:
1. **Training Phase**: Trains the specified RL agents for a set number of episodes and saves the checkpoints. It automatically skips training if a checkpoint already exists.
2. **Evaluation Phase**: Leverages Python multiprocessing (`forkserver`) to evaluate the agents across multiple Signal-to-Noise Ratios (SNRs). 
3. **Database Logging**: All evaluation progress is chunked (e.g., every 100 frames) and safely committed to a local, single-file DuckDB database (`experiments.db`).

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
> The evaluation loop is entirely interruptible! If the script crashes or you kill it, running it again will automatically pick up right where it left off, down to the exact SNR and chunk thanks to DuckDB.

---

## 2. Live Web Dashboard (Plotting)

Instead of generating hundreds of redundant `.png` files, all plotting and filtering is done dynamically in the browser. The dashboard fetches the latest data directly from `experiments.db` and renders Matplotlib charts in memory.

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
DuckDB is an embedded database. It runs inside the Python process and reads/writes to `experiments.db`. **You do not need to configure or request a SQL server from your cluster admins.**

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
