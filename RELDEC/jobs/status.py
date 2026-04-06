from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _pid_alive(pid: int | None) -> bool:
    if pid is None:
        return False
    try:
        os.kill(pid, 0)
        return True
    except Exception:
        return False


def _checkpoint_progress(checkpoint_latest: Path) -> dict[str, Any]:
    if not checkpoint_latest.exists():
        return {}
    try:
        with np.load(checkpoint_latest, allow_pickle=False) as npz:
            episodes = int(np.asarray(npz["episodes_completed"]).reshape(-1)[0])
            elapsed_sec = float(np.asarray(npz["elapsed_sec"]).reshape(-1)[0])
        return {"episodes_completed": episodes, "elapsed_sec": elapsed_sec}
    except Exception as exc:
        return {"error": str(exc)}


def _latest_eval_rows(latest_json: Path) -> int | None:
    if not latest_json.exists():
        return None
    try:
        payload = json.loads(latest_json.read_text(encoding="utf-8"))
        return len(payload.get("results", []))
    except Exception:
        return None


def _format_table(rows: list[list[str]], headers: list[str]) -> str:
    widths = [len(h) for h in headers]
    for row in rows:
        for i, col in enumerate(row):
            widths[i] = max(widths[i], len(col))

    def fmt(r: list[str]) -> str:
        return " | ".join(c.ljust(widths[i]) for i, c in enumerate(r))

    sep = "-+-".join("-" * w for w in widths)
    out = [fmt(headers), sep]
    for row in rows:
        out.append(fmt(row))
    return "\n".join(out)


def _main() -> None:
    ap = argparse.ArgumentParser(description="Show RELDEC job-run status")
    ap.add_argument("--run-root", required=True)
    ap.add_argument("--json", action="store_true", dest="as_json")
    args = ap.parse_args()

    run_root = Path(args.run_root)
    manifest = _load_json(run_root / "manifest.json")
    codes = list(manifest.get("codes", []))

    episodes_per_snr = int(manifest.get("train", {}).get("episodes_per_snr", 0))
    train_snr_count = int(manifest.get("train", {}).get("train_snr_count", 0))
    planned_episodes = episodes_per_snr * train_snr_count if train_snr_count else 0
    max_episodes = manifest.get("train", {}).get("max_episodes")
    if isinstance(max_episodes, int):
        total_episodes = min(planned_episodes, max_episodes) if planned_episodes else max_episodes
    else:
        total_episodes = planned_episodes

    fleet: dict[str, Any] = {
        "run_root": str(run_root),
        "created_at": manifest.get("created_at"),
        "codes": {},
    }

    rows: list[list[str]] = []
    for code in codes:
        train_state = _load_json(run_root / "state" / f"train_{code}.json")
        eval_state = _load_json(run_root / "state" / f"eval_{code}.json")

        train_pid = train_state.get("pid")
        train_alive = _pid_alive(int(train_pid)) if isinstance(train_pid, int) else False

        ckpt_latest = run_root / code / "checkpoints" / "checkpoint_latest.npz"
        progress = _checkpoint_progress(ckpt_latest)
        episodes_done = int(progress.get("episodes_completed", 0)) if progress else 0

        pct = 0.0
        if total_episodes > 0:
            pct = min(100.0, (episodes_done / total_episodes) * 100.0)

        berfer_rows = _latest_eval_rows(run_root / code / "results" / "latest_berfer.json")
        msg_rows = _latest_eval_rows(run_root / code / "results" / "latest_messages.json")

        train_status = str(train_state.get("status", "missing"))
        if train_alive:
            train_status = "running"

        eval_status = str(eval_state.get("status", "missing"))

        fleet["codes"][code] = {
            "train_status": train_status,
            "train_pid": train_pid,
            "episodes_completed": episodes_done,
            "episodes_target": total_episodes,
            "progress_pct": pct,
            "eval_status": eval_status,
            "latest_berfer_rows": berfer_rows,
            "latest_message_rows": msg_rows,
        }

        rows.append(
            [
                code,
                train_status,
                str(train_pid) if train_pid is not None else "-",
                f"{episodes_done}/{total_episodes}" if total_episodes else str(episodes_done),
                f"{pct:5.1f}%" if total_episodes else "-",
                eval_status,
                str(berfer_rows) if berfer_rows is not None else "-",
                str(msg_rows) if msg_rows is not None else "-",
            ]
        )

    if args.as_json:
        print(json.dumps(fleet, indent=2))
        return

    print(f"Run root: {run_root}")
    print(_format_table(
        rows,
        headers=["code", "train", "pid", "episodes", "progress", "eval", "berfer_rows", "msg_rows"],
    ))


if __name__ == "__main__":
    _main()
