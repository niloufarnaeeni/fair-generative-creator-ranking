import ast
from pathlib import Path
import shutil
import pandas as pd


RAW_DIR = Path(__file__).resolve().parents[3] / "data" / "kaito" / "raw"
CREATORS_CSV = str(RAW_DIR / "creators.csv")
CREATOR_DETAILS_CSV = str(RAW_DIR / "creator_details.csv")


def norm(x) -> str:
    return str(x).strip().lower()


def safe_list_parse(raw):
    if raw is None:
        return []
    s = str(raw).strip()
    if not s:
        return []
    try:
        v = ast.literal_eval(s)
        if isinstance(v, (list, tuple, set)):
            return [str(x).strip() for x in v if str(x).strip()]
        return [str(v).strip()] if str(v).strip() else []
    except Exception:
        s = s.strip("[]")
        parts = [p.strip().strip("'").strip('"') for p in s.split(",")]
        return [p for p in parts if p]


def main():
    creators_path = Path(CREATORS_CSV)
    details_path = Path(CREATOR_DETAILS_CSV)

    creators_df = pd.read_csv(creators_path)
    details_df = pd.read_csv(details_path)

    required_creators_cols = {"project_name", "creators"}
    required_details_cols = {"project_name", "creator_code", "Rank"}

    missing_1 = required_creators_cols - set(creators_df.columns)
    missing_2 = required_details_cols - set(details_df.columns)

    if missing_1:
        raise ValueError(f"creators.csv missing columns: {sorted(missing_1)}")
    if missing_2:
        raise ValueError(f"creator_details.csv missing columns: {sorted(missing_2)}")

    creators_df["project_name_norm"] = creators_df["project_name"].map(norm)
    details_df["project_name_norm"] = details_df["project_name"].map(norm)
    details_df["creator_code"] = details_df["creator_code"].astype(str).str.strip()
    details_df["Rank"] = pd.to_numeric(details_df["Rank"], errors="coerce")
    details_df = details_df.dropna(subset=["Rank", "creator_code", "project_name_norm"]).copy()

    details_df = details_df.sort_values(
        ["project_name_norm", "Rank", "creator_code"],
        ascending=[True, True, True],
    )

    project_to_creator_codes = (
        details_df.groupby("project_name_norm")["creator_code"]
        .apply(lambda s: list(dict.fromkeys(s.tolist())))
        .to_dict()
    )

    summary_rows = []
    new_creators_col = []

    for _, row in creators_df.iterrows():
        project_name = row["project_name"]
        pkey = row["project_name_norm"]

        old_creators = safe_list_parse(row["creators"])
        new_creators = project_to_creator_codes.get(pkey, [])

        new_creators_col.append(str(new_creators))

        summary_rows.append(
            {
                "project_name": project_name,
                "old_count": len(old_creators),
                "new_count": len(new_creators),
                "changed": old_creators != new_creators,
            }
        )

    summary_df = pd.DataFrame(summary_rows).sort_values(
        ["changed", "project_name"], ascending=[False, True]
    )

    print("\nSummary")
    print(summary_df.to_string(index=False))
    print(f"\nProjects changed: {int(summary_df['changed'].sum())}/{len(summary_df)}")

    backup_path = creators_path.with_suffix(".csv.bak")
    shutil.copy2(creators_path, backup_path)
    print(f"\nBackup written to: {backup_path}")

    creators_df["creators"] = new_creators_col
    creators_df = creators_df.drop(columns=["project_name_norm"])
    creators_df.to_csv(creators_path, index=False)

    print(f"Overwritten: {creators_path}")


if __name__ == "__main__":
    main()
