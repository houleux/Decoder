"""File-based catalog and query helpers for RELDEC runs and results.

This module intentionally stays SQL-free. It scans manifests and result files
on disk, normalizes the records, and exposes small query helpers that operate
on plain Python dictionaries.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Sequence


def _repo_root() -> Path:
    return Path(__file__).resolve().parent


def _candidate_run_dirs(root: Path | None = None) -> list[Path]:
    base = Path(root) if root is not None else _repo_root()
    return [base / "runs", base / "algorithms" / "runs"]


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {"results": payload}


def _coerce_scalar(value: Any) -> Any:
    if not isinstance(value, str):
        return value

    text = value.strip()
    if text == "":
        return ""

    lowered = text.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False

    try:
        if text.startswith(("0x", "-0x")):
            return int(text, 16)
        if any(ch in text for ch in (".", "e", "E")):
            return float(text)
        return int(text)
    except ValueError:
        return text


def _read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []

    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        return [{key: _coerce_scalar(value) for key, value in row.items()} for row in reader]


def _load_rows(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".json":
        payload = _read_json(path)
        rows = payload.get("results", payload)
        if isinstance(rows, list):
            return [dict(row) for row in rows if isinstance(row, dict)]
        return []
    if suffix == ".csv":
        return _read_csv(path)
    return []


def _stringify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _sequence_matches(sequence: Sequence[Any], value: str | None) -> bool:
    if value is None:
        return True
    return any(_stringify(item) == value for item in sequence)


@dataclass(frozen=True)
class RunRecord:
    """Normalized metadata for a training or evaluation run."""

    run_type: str
    run_id: str
    created_at_utc: str
    experiment: dict[str, Any]
    config: dict[str, Any]
    artifacts: dict[str, Any]
    manifest_path: Path
    root_dir: Path

    @property
    def code(self) -> str:
        return _stringify(self.experiment.get("code"))

    @property
    def policy_type(self) -> str:
        return _stringify(self.experiment.get("policy_type"))

    @property
    def methods(self) -> list[str]:
        methods = self.experiment.get("methods", [])
        if isinstance(methods, list):
            return [_stringify(method) for method in methods]
        return []

    @property
    def config_hash(self) -> str:
        return _stringify(self.artifacts.get("config_hash"))

    @property
    def to_row(self) -> dict[str, Any]:
        return {
            "run_type": self.run_type,
            "run_id": self.run_id,
            "created_at_utc": self.created_at_utc,
            "code": self.code,
            "policy_type": self.policy_type,
            "methods": list(self.methods),
            "config_hash": self.config_hash,
            "experiment": dict(self.experiment),
            "config": dict(self.config),
            "artifacts": dict(self.artifacts),
            "manifest_path": str(self.manifest_path),
            "root_dir": str(self.root_dir),
        }


@dataclass(frozen=True)
class ResultQuery:
    """Filter for file-based evaluation result rows."""

    code: str | None = None
    method: str | None = None
    run_id: str | None = None
    config_hash: str | None = None
    snr_db: float | None = None
    snr_min: float | None = None
    snr_max: float | None = None


class DataCatalog:
    """Scan RELDEC manifests and result files without using SQL."""

    def __init__(self, repo_root: str | Path | None = None):
        self.repo_root = Path(repo_root) if repo_root is not None else _repo_root()

    def _load_run_record(self, run_type: str, manifest_path: Path) -> RunRecord | None:
        payload = _read_json(manifest_path)
        if not payload:
            return None

        experiment = payload.get("experiment", {})
        config_key = "training_config" if run_type == "training" else "evaluation_config"
        return RunRecord(
            run_type=run_type,
            run_id=_stringify(payload.get("run_id")),
            created_at_utc=_stringify(payload.get("created_at_utc")),
            experiment=experiment if isinstance(experiment, dict) else {},
            config=payload.get(config_key, {}) if isinstance(payload.get(config_key, {}), dict) else {},
            artifacts=payload.get("artifacts", {}) if isinstance(payload.get("artifacts", {}), dict) else {},
            manifest_path=manifest_path,
            root_dir=manifest_path.parents[2] if len(manifest_path.parents) >= 3 else manifest_path.parent,
        )

    def iter_run_records(self, run_type: str | None = None) -> Iterator[RunRecord]:
        seen: set[tuple[str, str]] = set()
        for run_dir in _candidate_run_dirs(self.repo_root):
            if not run_dir.exists():
                continue

            types = [run_type] if run_type else ["training", "evaluation"]
            for current_type in types:
                if current_type not in {"training", "evaluation"}:
                    raise ValueError(f"Unknown run_type: {current_type}")

                for manifest_path in sorted((run_dir / current_type).glob("*/manifest.json")):
                    run_id = manifest_path.parent.name
                    key = (current_type, run_id)
                    if key in seen:
                        continue
                    seen.add(key)
                    record = self._load_run_record(current_type, manifest_path)
                    if record is not None:
                        yield record

    def query_runs(
        self,
        run_type: str | None = None,
        code: str | None = None,
        policy_type: str | None = None,
        run_id: str | None = None,
        config_hash: str | None = None,
        method: str | None = None,
    ) -> list[RunRecord]:
        records = []
        for record in self.iter_run_records(run_type=run_type):
            if code is not None and record.code != code:
                continue
            if policy_type is not None and record.policy_type != policy_type:
                continue
            if run_id is not None and record.run_id != run_id:
                continue
            if config_hash is not None and record.config_hash != config_hash:
                continue
            if method is not None and not _sequence_matches(record.methods, method):
                continue
            records.append(record)
        return records

    def iter_evaluation_rows(self, query: ResultQuery | None = None) -> Iterator[dict[str, Any]]:
        query = query or ResultQuery()
        for record in self.query_runs(
            run_type="evaluation",
            code=query.code,
            run_id=query.run_id,
            config_hash=query.config_hash,
            method=query.method,
        ):
            results_csv = record.artifacts.get("results_csv")
            results_json = record.artifacts.get("results_json")

            source_paths: list[Path] = []
            if isinstance(results_json, str) and results_json:
                source_paths.append(Path(results_json))
            if isinstance(results_csv, str) and results_csv:
                source_paths.append(Path(results_csv))

            rows: list[dict[str, Any]] = []
            for path in source_paths:
                rows = _load_rows(path)
                if rows:
                    break

            for row in rows:
                normalized = dict(row)
                normalized["run_id"] = record.run_id
                normalized["created_at_utc"] = record.created_at_utc
                normalized["config_hash"] = record.config_hash
                normalized["code"] = record.code
                normalized["matrix_csv"] = record.experiment.get("matrix_csv")
                normalized["methods"] = list(record.methods)
                normalized["results_source"] = str(source_paths[0]) if source_paths else ""

                if query.method is not None and _stringify(normalized.get("method")) != query.method:
                    continue

                snr_value = normalized.get("snr_db")
                if query.snr_db is not None and float(snr_value) != float(query.snr_db):
                    continue
                if query.snr_min is not None and float(snr_value) < float(query.snr_min):
                    continue
                if query.snr_max is not None and float(snr_value) > float(query.snr_max):
                    continue

                yield normalized

    def query_evaluation_rows(self, query: ResultQuery | None = None) -> list[dict[str, Any]]:
        return list(self.iter_evaluation_rows(query=query))


def rows_to_csv(rows: Sequence[dict[str, Any]], output_path: str | Path | None = None) -> str:
    if not rows:
        return ""

    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)

    if output_path is None:
        from io import StringIO

        buffer = StringIO()
        writer = csv.DictWriter(buffer, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        return buffer.getvalue()

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return str(path)


def rows_to_json(rows: Sequence[dict[str, Any]], output_path: str | Path | None = None) -> str:
    payload = json.dumps(list(rows), indent=2)
    if output_path is None:
        return payload

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload, encoding="utf-8")
    return str(path)
