import duckdb
import json

conn = duckdb.connect('experiments.db')

# z=4 rows
rows = conn.execute("""
    SELECT c.config_json, e.snr_db, e.frames_done, e.max_frames
    FROM eval_results e 
    JOIN configs c ON e.config_id = c.config_id 
    WHERE c.config_json LIKE '%"z": 4%'
""").fetchall()
print(f"--- z=4 Eval Rows: {len(rows)} ---")
for r in rows:
    cfg = json.loads(r[0])
    print(f"{cfg['method']} z=4, SNR={r[1]}, frames_done: {r[2]}/{r[3]}")

# mackay rows
m_rows = conn.execute("""
    SELECT c.config_json, e.snr_db, e.frames_done, e.max_frames
    FROM eval_results e 
    JOIN configs c ON e.config_id = c.config_id 
    WHERE c.config_json LIKE '%mackay%' OR c.config_json LIKE '%MacKay%'
""").fetchall()
print(f"\n--- MacKay Eval Rows: {len(m_rows)} ---")
for r in m_rows:
    cfg = json.loads(r[0])
    print(f"{cfg['method']} MacKay, SNR={r[1]}, frames_done: {r[2]}/{r[3]}")
