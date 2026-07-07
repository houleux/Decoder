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

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/configs')
def get_configs():
    conn = get_conn()
    rows = conn.execute("SELECT config_id, created_at, description, tags, config_json FROM configs ORDER BY created_at DESC").fetchall()
    
    configs = []
    for row in rows:
        config_data = json.loads(row[4])
        configs.append({
            "id": row[0],
            "created_at": row[1].isoformat(),
            "description": row[2],
            "tags": row[3],
            "data": config_data
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

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
