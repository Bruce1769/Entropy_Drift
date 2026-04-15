import argparse
import json
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd


SUPPORTED_EXTENSIONS = {".csv", ".jsonl", ".json"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export routed token positions from token-level recorder outputs."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to a token-level recorder file or a directory of such files.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output path. Uses .csv or .jsonl based on extension.",
    )
    parser.add_argument(
        "--sample-id-col",
        default=None,
        help=(
            "Column used as sample identifier. If omitted, auto-detect from "
            "`sample_id`, `problem_id`, `batch_id`, `rid`; otherwise fall back to filename."
        ),
    )
    parser.add_argument(
        "--source-col",
        default="source_model",
        help="Column containing quick/reference labels.",
    )
    parser.add_argument(
        "--position-col",
        default="position",
        help="Column containing decode positions.",
    )
    parser.add_argument(
        "--reference-label",
        default="reference",
        help="Value in source column that means the token was routed to the large model.",
    )
    parser.add_argument(
        "--token-col",
        default="token_str",
        help="Optional token text column. Exported only if present.",
    )
    return parser.parse_args()


def iter_input_files(input_path: Path) -> Iterable[Path]:
    if input_path.is_file():
        yield input_path
        return

    if not input_path.is_dir():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    for path in sorted(input_path.iterdir()):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            yield path


def load_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".jsonl":
        return pd.read_json(path, lines=True)
    if suffix == ".json":
        data = json.loads(path.read_text())
        if isinstance(data, list):
            return pd.DataFrame(data)
        if isinstance(data, dict):
            if "records" in data and isinstance(data["records"], list):
                return pd.DataFrame(data["records"])
            raise ValueError(f"Unsupported JSON object format in {path}")
    raise ValueError(f"Unsupported input file format: {path}")


def detect_sample_id_col(df: pd.DataFrame, preferred: Optional[str]) -> Optional[str]:
    if preferred:
        if preferred not in df.columns:
            raise ValueError(f"Requested sample id column `{preferred}` not found.")
        return preferred

    for name in ("sample_id", "problem_id", "batch_id", "rid"):
        if name in df.columns:
            return name
    return None


def export_routes_for_dataframe(
    df: pd.DataFrame,
    file_id: str,
    sample_id_col: Optional[str],
    source_col: str,
    position_col: str,
    reference_label: str,
    token_col: str,
) -> list[dict]:
    missing = [col for col in (source_col, position_col) if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns {missing} in {file_id}")

    df = df.copy()
    group_col = sample_id_col or "__sample_id__"
    if sample_id_col is None:
        df[group_col] = file_id

    routed = df[df[source_col] == reference_label]
    rows = []
    for sample_id, group in routed.groupby(group_col, dropna=False):
        group = group.sort_values(position_col)
        row = {
            "sample_id": sample_id,
            "route_count": int(len(group)),
            "route_positions": [int(x) for x in group[position_col].tolist()],
        }
        if token_col in group.columns:
            row["route_tokens"] = [
                "" if pd.isna(x) else str(x) for x in group[token_col].tolist()
            ]
        rows.append(row)

    return rows


def save_rows(rows: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = output_path.suffix.lower()
    if suffix == ".csv":
        df = pd.DataFrame(rows)
        for col in ("route_positions", "route_tokens"):
            if col in df.columns:
                df[col] = df[col].apply(
                    lambda x: json.dumps(x, ensure_ascii=False) if isinstance(x, list) else x
                )
        df.to_csv(output_path, index=False)
        return

    if suffix == ".jsonl":
        with output_path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        return

    raise ValueError("Output file must end with .csv or .jsonl")


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    all_rows: list[dict] = []

    for path in iter_input_files(input_path):
        df = load_table(path)
        sample_id_col = detect_sample_id_col(df, args.sample_id_col)
        file_rows = export_routes_for_dataframe(
            df=df,
            file_id=path.stem,
            sample_id_col=sample_id_col,
            source_col=args.source_col,
            position_col=args.position_col,
            reference_label=args.reference_label,
            token_col=args.token_col,
        )
        all_rows.extend(file_rows)

    save_rows(all_rows, Path(args.output))
    print(f"Exported {len(all_rows)} samples to {args.output}")


if __name__ == "__main__":
    main()
