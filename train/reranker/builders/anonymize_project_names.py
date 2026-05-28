from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import pandas as pd


RAW_DIR = Path(__file__).resolve().parents[3] / "data" / "kaito" / "raw"
DEFAULT_FILES = [
    RAW_DIR / "project_descriptions.csv",
    RAW_DIR / "extended_project_description.csv",
]


def _normalize_name(value: str) -> str:
    return str(value).strip()


def _collect_project_names(csv_paths: List[Path]) -> List[str]:
    names = set()
    for csv_path in csv_paths:
        df = pd.read_csv(csv_path)
        if "project_name" not in df.columns:
            raise KeyError(f"{csv_path} is missing a 'project_name' column.")
        for value in df["project_name"].dropna().tolist():
            normalized = _normalize_name(value)
            if normalized:
                names.add(normalized)
    return sorted(names, key=str.lower)


def _build_name_mapping(project_names: List[str]) -> Dict[str, str]:
    return {
        project_name: f"P_{idx}"
        for idx, project_name in enumerate(project_names, start=1)
    }


def _replace_project_mentions(text: str, mapping: Dict[str, str]) -> str:
    updated = str(text)
    for project_name in sorted(mapping.keys(), key=len, reverse=True):
        updated = updated.replace(project_name, mapping[project_name])
    return updated


def anonymize_csv(csv_path: Path, mapping: Dict[str, str]) -> int:
    df = pd.read_csv(csv_path)
    original_names = df["project_name"].astype(str).map(_normalize_name)
    df["project_name"] = original_names.map(mapping)

    if "project_description" in df.columns:
        df["project_description"] = df["project_description"].map(
            lambda value: _replace_project_mentions(value, mapping)
            if pd.notna(value)
            else value
        )

    df.to_csv(csv_path, index=False)
    return len(df)


def write_mapping_csv(mapping: Dict[str, str], output_path: Path) -> None:
    rows = [
        {"project_name": project_name, "anonymous_project_name": anonymous_name}
        for project_name, anonymous_name in mapping.items()
    ]
    pd.DataFrame(rows).to_csv(output_path, index=False)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Anonymize project_name values across the Kaito project-description CSV files."
    )
    parser.add_argument(
        "--files",
        nargs="*",
        default=[str(path) for path in DEFAULT_FILES],
        help="CSV files whose project_name column should be anonymized in place.",
    )
    parser.add_argument(
        "--mapping_out",
        default=str(DEFAULT_FILES[0].parent / "project_name_mapping.csv"),
        help="Path to save the original-to-anonymized project name mapping.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    csv_paths = [Path(path) for path in args.files]

    project_names = _collect_project_names(csv_paths)
    mapping = _build_name_mapping(project_names)

    total_rows = 0
    for csv_path in csv_paths:
        row_count = anonymize_csv(csv_path, mapping)
        total_rows += row_count
        print(f"[Anonymized] rows={row_count} path={csv_path}")

    mapping_out = Path(args.mapping_out)
    write_mapping_csv(mapping, mapping_out)
    print(f"[Mapping] projects={len(mapping)} path={mapping_out}")
    print(f"[Done] files={len(csv_paths)} total_rows={total_rows}")


if __name__ == "__main__":
    main()
