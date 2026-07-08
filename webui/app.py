from flask import Flask, render_template, jsonify, request
import sys
import os
import json
import io
import base64
import matplotlib
matplotlib.use('Agg') # Headless rendering
import matplotlib.pyplot as plt

# Ensure we can import expdb
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from expdb.db import get_conn
import glob
import subprocess

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/configs')
def get_configs():
    conn = get_conn()
    query = """
        SELECT c.config_id, c.created_at, c.description, c.tags, c.config_json,
               COALESCE(MAX(e.frames_done), 0) as max_frames_done
        FROM configs c
        LEFT JOIN eval_results e ON c.config_id = e.config_id
        GROUP BY c.config_id, c.created_at, c.description, c.tags, c.config_json
        ORDER BY c.created_at DESC
    """
    rows = conn.execute(query).fetchall()
    
    configs = []
    for row in rows:
        config_data = json.loads(row[4])
        configs.append({
            "id": row[0],
            "created_at": row[1].isoformat(),
            "description": row[2],
            "tags": row[3],
            "data": config_data,
            "frames_done": row[5]
        })
    return jsonify(configs)

@app.route('/api/configs/<config_id>')
def get_config_details(config_id):
    conn = get_conn()
    
    # Get config
    row = conn.execute("SELECT config_json, description, tags FROM configs WHERE config_id = ?", (config_id,)).fetchone()
    if not row:
        return jsonify({"error": "Config not found"}), 404
        
    config_dict = json.loads(row[0])
    
    # Get runs
    run_rows = conn.execute("SELECT run_id, run_type, status, episodes_done, completed_at FROM runs WHERE config_id = ?", (config_id,)).fetchall()
    runs = []
    for r in run_rows:
        runs.append({
            "run_id": r[0],
            "run_type": r[1],
            "status": r[2],
            "episodes_done": r[3],
            "completed_at": r[4].isoformat() if r[4] else None
        })
        
    # Get evaluations
    eval_rows = conn.execute("""
        SELECT snr_db, target_frame_errors, max_frames, frames_done, completed, bit_errors, total_bits, frame_errors, messages 
        FROM eval_results 
        WHERE config_id = ? 
        ORDER BY target_frame_errors, max_frames, snr_db
    """, (config_id,)).fetchall()
    
    evals = []
    for e in eval_rows:
        snr, tfe, mf, done, comp, be, tb, fe, msg = e
        ber = be / tb if tb > 0 else 0
        fer = fe / done if done > 0 else 0
        avg_msg = msg / done if done > 0 else 0
        
        evals.append({
            "snr_db": snr,
            "target_frame_errors": tfe,
            "max_frames": mf,
            "frames_done": done,
            "completed": comp,
            "ber": ber,
            "fer": fer,
            "avg_messages": avg_msg
        })
        
    return jsonify({
        "id": config_id,
        "config": config_dict,
        "runs": runs,
        "evals": evals
    })

@app.route('/api/plot', methods=['POST'])
def plot_configs():
    data = request.json
    config_ids = data.get('config_ids', [])
    
    if not config_ids:
        return jsonify({"error": "No config IDs provided"}), 400
        
    conn = get_conn()
    placeholders = ','.join(['?'] * len(config_ids))
    
    # Get configs for titles/legends
    config_rows = conn.execute(f"SELECT config_id, config_json FROM configs WHERE config_id IN ({placeholders})", config_ids).fetchall()
    config_titles = {}
    for row in config_rows:
        cfg = json.loads(row[1])
        method = cfg.get("method", "Unknown")
        z = cfg.get("z", "?")
        config_titles[row[0]] = f"{method} (z={z})"
        
    # Get evaluations
    eval_rows = conn.execute(f"""
        SELECT config_id, snr_db, target_frame_errors, max_frames, frames_done, completed, bit_errors, total_bits, frame_errors, messages 
        FROM eval_results 
        WHERE config_id IN ({placeholders})
        ORDER BY target_frame_errors, max_frames, snr_db
    """, config_ids).fetchall()
    
    # Group by config_id
    grouped_data = {}
    for cid in config_ids:
        grouped_data[cid] = {}
        
    for e in eval_rows:
        cid, snr, tfe, mf, done, comp, be, tb, fe, msg = e
        if done == 0: continue
        
        ber = be / tb if tb > 0 else 0
        fer = fe / done if done > 0 else 0
        avg_msg = msg / done if done > 0 else 0
        
        if snr not in grouped_data[cid] or grouped_data[cid][snr]['done'] < done:
            grouped_data[cid][snr] = {'ber': ber, 'fer': fer, 'msg': avg_msg, 'done': done}
            
    # Create Matplotlib Figure
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
    
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']
    
    for i, cid in enumerate(config_ids):
        title = config_titles.get(cid, str(cid)[:8])
        marker = markers[i % len(markers)]
        
        snrs = sorted(grouped_data[cid].keys())
        if not snrs:
            continue
            
        bers = [grouped_data[cid][snr]['ber'] for snr in snrs]
        fers = [grouped_data[cid][snr]['fer'] for snr in snrs]
        msgs = [grouped_data[cid][snr]['msg'] for snr in snrs]
        
        ax1.plot(snrs, bers, label=title, marker=marker, markersize=5)
        ax2.plot(snrs, fers, label=title, marker=marker, markersize=5)
        ax3.plot(snrs, msgs, label=title, marker=marker, markersize=5)

    # Style axes based on codebase plot_sweep.py conventions
    for ax, ylabel, is_log in zip([ax1, ax2, ax3], ["BER", "FER", "Average Messages"], [True, True, False]):
        ax.set_xlabel("Eb/N0 (dB)")
        ax.set_ylabel(ylabel)
        if is_log:
            ax.set_yscale("log")
        ax.grid(True, which="both", ls="-", alpha=0.5)
        
    ax1.legend(fontsize="small", loc="lower left")
    
    plt.tight_layout()
    
    # Save to base64
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    buf.seek(0)
    img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    
    return jsonify({
        "image": f"data:image/png;base64,{img_base64}"
    })

@app.route('/api/matrices', methods=['GET'])
def get_matrices():
    matrices = glob.glob(os.path.join("..", "matrices", "*.csv"))
    # The current working directory for app.py is often Decoder/, but if it's Decoder/webui, we handle it.
    # Actually, sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    # The CWD where webui/app.py is run is usually the root `Decoder`.
    # Let's glob absolute path just to be safe.
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    matrix_dir = os.path.join(base_dir, "matrices")
    matrices = glob.glob(os.path.join(matrix_dir, "*.csv"))
    
    # Return relative paths from root, like "matrices/H_AB_LDPC_500.csv"
    rel_matrices = [os.path.relpath(m, base_dir) for m in matrices]
    return jsonify({"matrices": sorted(rel_matrices)})

@app.route('/api/methods', methods=['GET'])
def get_methods():
    methods = [
        "flooding", "round_robin", "random", "rbl", "ave_rbl", "max_rbl",
        "reldec", "dyna_reldec", "ave_res_q", "max_res_q",
        "llr_vec_ave_res", "llr_vec_ave_mi", "ave_llr_ave_res", "ave_llr_ave_mi",
        "tanh_vec_ave_res", "tanh_vec_ave_mi", "ave_tanh_ave_res", "ave_tanh_ave_mi",
        "ave_mi_ave_mi"
    ]
    return jsonify({"methods": methods})

@app.route('/api/run_experiment', methods=['POST'])
def run_experiment():
    data = request.json
    
    # Build command
    cmd = ["python3", "run_experiments.py"]
    
    if "matrix" in data and data["matrix"]:
        cmd.extend(["--matrix", data["matrix"]])
    else:
        return jsonify({"error": "Matrix is required"}), 400
        
    if "methods" in data and data["methods"]:
        cmd.extend(["--methods"] + data["methods"])
    else:
        return jsonify({"error": "At least one method is required"}), 400
        
    if "zVals" in data and data["zVals"]:
        cmd.extend(["--z-vals"] + data["zVals"].split())
        
    if "trainSnrs" in data and data["trainSnrs"]:
        cmd.extend(["--train-snrs"] + data["trainSnrs"].split())
        
    if "evalSnrs" in data and data["evalSnrs"]:
        cmd.extend(["--eval-snrs"] + data["evalSnrs"].split())
        
    if "trainEpisodes" in data and data["trainEpisodes"]:
        cmd.extend(["--train-episodes", str(data["trainEpisodes"])])
        
    if "maxFrames" in data and data["maxFrames"]:
        cmd.extend(["--max-frames", str(data["maxFrames"])])
        
    if "workers" in data and data["workers"]:
        cmd.extend(["--workers", str(data["workers"])])
        
        
    if "lMax" in data and data["lMax"]:
        cmd.extend(["--l-max", str(data["lMax"])])
        
    if "seed" in data and data["seed"]:
        cmd.extend(["--seed", str(data["seed"])])
        
    if "targetFrameErrors" in data and data["targetFrameErrors"]:
        cmd.extend(["--target-frame-errors", str(data["targetFrameErrors"])])
        
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    
    try:
        # Spawn process in background
        proc = subprocess.Popen(
            cmd,
            cwd=base_dir,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        return jsonify({
            "message": "Experiment started successfully",
            "pid": proc.pid,
            "command": " ".join(cmd)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
