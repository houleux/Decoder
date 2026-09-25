"""Validate matlab_bridge crash-safety/idempotency against a TEMP database."""
import atexit, json, shutil, sys, tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from expdb.db import temporary_database

# A throwaway database next to the production one, dropped on exit (including
# on a crash part-way through).
_tmp_db = temporary_database(prefix="bridge_test_")
print("temp db:", _tmp_db.__enter__())
atexit.register(_tmp_db.__exit__, None, None, None)
tmpdir = tempfile.mkdtemp(prefix="bridge_test_")

from matlab_bridge.ingest import ingest_once
from matlab_bridge.config import config_from_manifest, enforce_matlab_prefix, MethodPrefixError
from matlab_bridge.spool import load_ledger, read_manifest, SENTINEL
from expdb import get_or_create_config, query_ber

spool = Path(tmpdir) / "spool"; spool.mkdir()
manifest = {
    "method":"matlab_reldec_quartile_k3","matrix":"matrices/WRAN_irreg_384_256.csv",
    "matrix_matlab":"x.mat","z":1,"k":3,"state_encoding":"quartile_avg_llr",
    "alpha":0.1,"gamma":0.9,"epsilon":0.1,"l_max":50,"train_episodes":500,
    "train_snr_vals":0.0,"seed":42,"eval_maxIter":5,"eval_base_seed":12345,"scheduling":"sweep",
    "target_frame_errors":15,"max_frames":15,"checkpoint":"c.mat","n_shards":4,
    "chunk_size":5,"interpreter":"test","source":"test",
}
(spool/"manifest.json").write_text(json.dumps(manifest))

def chunk(cid, snr, frames, be, fe):
    return {"chunk_id":cid,"sentinel":SENTINEL,"method":manifest["method"],"snr_db":snr,
            "target_frame_errors":15,"max_frames":15,"frames":frames,"bit_errors":be,
            "total_bits":frames*384,"frame_errors":fe,"messages":frames*1000,
            "converged_frames":0,"substream":1,"wall_s":1.0}

for i,(snr,be,fe) in enumerate([(1.0,100,5),(1.0,120,5),(2.0,80,5),(2.0,90,5)]):
    (spool/f"c{i}.json").write_text(json.dumps(chunk(f"c{i}",snr,5,be,fe)))
# a deliberately torn file: valid-looking name, truncated content
(spool/"c_partial.json").write_text('{"chunk_id":"c_partial","frames":5')

cfg = config_from_manifest(manifest)
cid = get_or_create_config(cfg)
print("config_id:", cid, "\nconfig:", json.dumps(cfg, sort_keys=True))

ing = load_ledger(spool)
n, d = ingest_once(spool, cid, manifest, ing, verbose=False)
print(f"\npass1: committed={n} deferred={d}  (expect 4 committed, 1 deferred)")
rows = {r["snr_db"]:r for r in query_ber(cid,15,15)}
for s,r in sorted(rows.items()):
    print(f"   snr={s}: frames={r['frames_done']} ber={r['ber']:.6f} fer={r['fer']:.4f}")

# ---- simulate ingester restart on the SAME spool: must not double count ----
ing2 = load_ledger(spool)
n2, d2 = ingest_once(spool, cid, manifest, ing2, verbose=False)
rows2 = {r["snr_db"]:r for r in query_ber(cid,15,15)}
print(f"\npass2 (restart): committed={n2} (expect 0)")
same = all(rows[s]["frames_done"]==rows2[s]["frames_done"] for s in rows)
print(f"   frame counts unchanged after restart: {same}")
print(f"   snr=1.0 frames still {rows2[1.0]['frames_done']} (expect 10, NOT 20)")

# ---- prefix enforcement ----
print("\nprefix enforcement:")
print("   'matlab_x' ->", enforce_matlab_prefix("matlab_x"))
for bad in ["reldec", "", None]:
    try:
        enforce_matlab_prefix(bad); print(f"   {bad!r} -> NO ERROR  <-- BUG")
    except MethodPrefixError:
        print(f"   {bad!r} -> correctly rejected")

ok = (n==4 and d==1 and n2==0 and same and rows2[1.0]['frames_done']==10)
print("\nRESULT:", "PASS" if ok else "FAIL")
shutil.rmtree(tmpdir)
sys.exit(0 if ok else 1)
