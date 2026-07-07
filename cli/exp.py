import argparse
import sys
import json
import os
import pandas as pd
from tabulate import tabulate

from expdb import get_or_create_config, create_run, update_run_status, query_ber, get_coverage
from expdb.db import get_conn

def cmd_show(args):
    if args.config_id:
        config_id = args.config_id
    else:
        # Load from file if --config is passed, otherwise we need to reconstruct from args...
        # For simplicity, if config_id isn't provided, tell user to use it.
        print("Please provide --config-id")
        return
        
    conn = get_conn()
    row = conn.execute("SELECT config_json, description, tags FROM configs WHERE config_id = ?", (config_id,)).fetchone()
    if not row:
        print(f"Config {config_id} not found.")
        return
        
    print(f"=== Config: {config_id} ===")
    if row[1]: print(f"Description: {row[1]}")
    if row[2]: print(f"Tags: {row[2]}")
    
    config_dict = json.loads(row[0])
    print("\nParameters:")
    for k, v in config_dict.items():
        print(f"  {k}: {v}")
        
    runs = conn.execute("SELECT run_id, run_type, status, episodes_done, completed_at FROM runs WHERE config_id = ?", (config_id,)).fetchall()
    print("\nRuns:")
    for r in runs:
        print(f"  {r[0]} | {r[1]} | {r[2]} | eps: {r[3]} | {r[4]}")
        
    print("\nEvaluations:")
    evals = conn.execute("SELECT snr_db, target_frame_errors, max_frames, frames_done, completed, bit_errors, total_bits, frame_errors, messages FROM eval_results WHERE config_id = ? ORDER BY target_frame_errors, max_frames, snr_db", (config_id,)).fetchall()
    
    if evals:
        headers = ["SNR", "Target FE", "Max Frames", "Done", "Completed", "BER", "FER", "Avg Msg"]
        table = []
        for e in evals:
            snr, tfe, mf, done, comp, be, tb, fe, msg = e
            ber = be/tb if tb > 0 else 0
            fer = fe/done if done > 0 else 0
            avg_msg = msg/done if done > 0 else 0
            table.append([snr, tfe, mf, done, comp, f"{ber:.5e}", f"{fer:.5e}", f"{avg_msg:.2f}"])
        print(tabulate(table, headers=headers))
    else:
        print("  No evaluations found.")

def cmd_ls(args):
    conn = get_conn()
    query = "SELECT config_id, created_at, description, tags FROM configs"
    params = []
    
    if args.tag:
        query += " WHERE tags LIKE ?"
        params.append(f"%{args.tag}%")
        
    rows = conn.execute(query, params).fetchall()
    
    headers = ["Config ID", "Created At", "Description", "Tags"]
    print(tabulate(rows, headers=headers))

def main():
    parser = argparse.ArgumentParser(description="Experiment Management CLI")
    subparsers = parser.add_subparsers(dest="command")
    
    show_parser = subparsers.add_parser("show", help="Show details for a config")
    show_parser.add_argument("--config-id", type=str, required=True, help="Config ID")
    
    ls_parser = subparsers.add_parser("ls", help="List configs")
    ls_parser.add_argument("--tag", type=str, help="Filter by tag")
    
    args = parser.parse_args()
    
    if args.command == "show":
        cmd_show(args)
    elif args.command == "ls":
        cmd_ls(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
