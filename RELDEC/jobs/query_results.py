from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from RELDEC.data_catalog import DataCatalog, ResultQuery, rows_to_csv, rows_to_json


def _format_table(rows: list[dict[str, object]], fields: list[str]) -> str:
    if not rows:
        return "No matching rows found."

    widths = [len(field) for field in fields]
    rendered_rows: list[list[str]] = []
    for row in rows:
        values = [str(row.get(field, "")) for field in fields]
        rendered_rows.append(values)
        for idx, value in enumerate(values):
            widths[idx] = max(widths[idx], len(value))

    def fmt(values: list[str]) -> str:
        return " | ".join(value.ljust(widths[idx]) for idx, value in enumerate(values))

    separator = "-+-".join("-" * width for width in widths)
    lines = [fmt(fields), separator]
    lines.extend(fmt(values) for values in rendered_rows)
    return "\n".join(lines)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Query RELDEC run manifests and evaluation tables without SQL.")
    parser.add_argument("--kind", choices=["training", "evaluation"], default="evaluation")
    parser.add_argument("--code", type=str, default=None)
    parser.add_argument("--policy-type", type=str, default=None)
    parser.add_argument("--method", type=str, default=None)
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--config-hash", type=str, default=None)
    parser.add_argument("--snr-db", type=float, default=None)
    parser.add_argument("--snr-min", type=float, default=None)
    parser.add_argument("--snr-max", type=float, default=None)
    parser.add_argument("--format", choices=["table", "json", "csv"], default="table")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--fields", nargs="+", default=None)
    parser.add_argument("--output", type=str, default=None, help="Optional file to write JSON/CSV output to")
    return parser.parse_args()


def _default_fields(kind: str) -> list[str]:
    if kind == "training":
        return ["run_id", "created_at_utc", "code", "policy_type", "config_hash"]
    return ["run_id", "created_at_utc", "code", "method", "snr_db", "ber", "fer", "avg_messages", "avg_iterations"]


def _main() -> None:
    args = _parse_args()
    catalog = DataCatalog()

    if args.kind == "training":
        records = catalog.query_runs(
            run_type="training",
            code=args.code,
            policy_type=args.policy_type,
            run_id=args.run_id,
            config_hash=args.config_hash,
            method=args.method,
        )
        rows = [record.to_row for record in records]
    else:
        query = ResultQuery(
            code=args.code,
            method=args.method,
            run_id=args.run_id,
            config_hash=args.config_hash,
            snr_db=args.snr_db,
            snr_min=args.snr_min,
            snr_max=args.snr_max,
        )
        rows = catalog.query_evaluation_rows(query)

    if args.limit is not None:
        rows = rows[: max(args.limit, 0)]

    fields = args.fields or _default_fields(args.kind)

    if args.format == "json":
        payload = rows_to_json(rows, args.output)
        if args.output is None:
            print(payload)
        else:
            print(payload)
        return

    if args.format == "csv":
        payload = rows_to_csv(rows, args.output)
        if args.output is None:
            sys.stdout.write(payload)
        else:
            print(payload)
        return

    if args.output is not None:
        output_path = Path(args.output)
        if output_path.suffix.lower() == ".json":
            rows_to_json(rows, output_path)
        else:
            rows_to_csv(rows, output_path)

    print(_format_table(rows, fields))


if __name__ == "__main__":
    _main()
