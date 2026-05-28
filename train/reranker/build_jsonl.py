import argparse
import ast
import json
import logging
import os
import pickle
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple
import sys

import pandas as pd

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parent))
    from builders.preprocess import preprocess
else:
    from .builders.preprocess import preprocess


log = logging.getLogger(__name__)
TEMPLATE_DIR = Path(__file__).resolve().parent / "templates"
TOP_PROMPT_TEMPLATE_PATH = TEMPLATE_DIR / "top_prompt.txt"
RELEVANCE_PROMPT_TEMPLATE_PATH = TEMPLATE_DIR / "relevance_prompt.txt"
EXACT_RANGE_PROMPT_TEMPLATE_PATH = TEMPLATE_DIR / "exact_range_prompt.txt"

TOP_TARGET_SPECS = [
    {"target_type": "top_5", "start_rank": 1, "end_rank": 5, "k": 5},
    {"target_type": "top_10", "start_rank": 1, "end_rank": 10, "k": 10},
    {"target_type": "top_20", "start_rank": 1, "end_rank": 20, "k": 20},
]

TEST_TARGET_SPECS = [
    {"target_type": "test_relevance_5", "k": 5},
    {"target_type": "test_relevance_10", "k": 10},
    {"target_type": "test_relevance_20", "k": 20},
]


def _norm_key(x: Any) -> str:
    return str(x).strip().lower()


def _safe_list_parse(raw: Any) -> List[str]:
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
        s2 = s.strip().strip("[]")
        parts = [p.strip().strip("'").strip('"') for p in s2.split(",")]
        return [p for p in parts if p]


def t_from_rank(rank: int, n_pos: int) -> float:
    if n_pos <= 0:
        raise ValueError("n_pos must be >= 1")
    if not (1 <= rank <= n_pos):
        raise ValueError(f"rank out of range: rank={rank}, n_pos={n_pos}")
    return (n_pos - rank + 1) / n_pos


def _load_prompt_template(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Missing prompt template: {path}")
    return path.read_text(encoding="utf-8").strip()


def _load_splits_bundle(splits_pkl: str) -> Dict:
    with open(splits_pkl, "rb") as f:
        obj = pickle.load(f)

    if isinstance(obj, dict) and "trials" in obj:
        return obj

    return {"trials": {0: obj}, "n_trials": 1}


def _load_indexes(indexes_pkl: str) -> Dict:
    with open(indexes_pkl, "rb") as f:
        return pickle.load(f)


def _maybe_load_teamsvecs_shape(teamsvecs_pkl: str) -> Optional[Tuple[int, int]]:
    try:
        with open(teamsvecs_pkl, "rb") as f:
            tv = pickle.load(f)
        n_teams = tv["skill"].shape[0]
        n_members = tv["member"].shape[1]
        return n_teams, n_members
    except Exception:
        return None


def _row_indices_to_project_names(
    row_indices: List[int],
    indexes: Dict,
    skills_df: pd.DataFrame,
) -> List[str]:
    if "i2t" not in indexes:
        raise KeyError("indexes.pkl is missing 'i2t' (row_idx -> team_id mapping).")

    if "project_id" not in skills_df.columns or "project_name" not in skills_df.columns:
        raise KeyError("skills.csv must contain columns: project_id, project_name")

    pid_to_name = {}
    for _, r in skills_df.iterrows():
        try:
            pid = int(r["project_id"])
            pname = str(r["project_name"]).strip()
            if pname:
                pid_to_name[pid] = pname
        except Exception:
            continue

    out = []
    missing = 0
    for idx in row_indices:
        idx = int(idx)

        if isinstance(indexes["i2t"], dict):
            if idx not in indexes["i2t"]:
                missing += 1
                continue
            project_id = int(indexes["i2t"][idx])
        else:
            project_id = int(indexes["i2t"][idx])

        pname = pid_to_name.get(project_id)
        if not pname:
            missing += 1
            continue
        out.append(pname)

    if missing:
        log.warning(
            "[Mapping] %s split indices could not be mapped to a project_name (skipped).",
            missing,
        )

    seen = set()
    deduped = []
    for x in out:
        k = _norm_key(x)
        if k not in seen:
            seen.add(k)
            deduped.append(x)
    return deduped


def load_creator_yaps(raw_dir: Path) -> Dict[str, float]:
    yaps_file = raw_dir / "yap_scores.csv"
    creators_file = raw_dir / "creator_details.csv"

    if not yaps_file.exists():
        raise FileNotFoundError(f"Missing Yaps file: {yaps_file}")

    if not creators_file.exists():
        raise FileNotFoundError(f"Missing creator details file: {creators_file}")

    yaps_df = pd.read_csv(yaps_file)
    creators_df = pd.read_csv(creators_file)

    yaps_df["username_norm"] = yaps_df["username"].astype(str).str.lower().str.lstrip("@")
    creators_df["creator_id_norm"] = (
        creators_df["Creator_ID"].astype(str).str.lower().str.lstrip("@")
    )

    merged = creators_df.merge(
        yaps_df,
        left_on="creator_id_norm",
        right_on="username_norm",
        how="left",
    )

    return {
        str(row["creator_code"]): float(row["yaps_all"])
        for _, row in merged.iterrows()
        if not pd.isna(row["yaps_all"])
    }


def _load_project_name_aliases(raw_dir: Path) -> Dict[str, str]:
    mapping_path = raw_dir / "project_name_mapping.csv"
    if not mapping_path.exists():
        return {}

    df = pd.read_csv(mapping_path)
    required_columns = {"project_name", "anonymous_project_name"}
    if not required_columns.issubset(df.columns):
        raise KeyError(
            f"{mapping_path} must contain columns: {sorted(required_columns)}"
        )

    aliases: Dict[str, str] = {}
    for _, row in df.iterrows():
        original = _norm_key(row["project_name"])
        anonymized = _norm_key(row["anonymous_project_name"])
        if not original or not anonymized:
            continue
        aliases[original] = anonymized
        aliases[anonymized] = original
    return aliases


def _load_project_description_variants(raw_dir: Path) -> Dict[str, List[str]]:
    description_map: Dict[str, List[str]] = {}

    candidate_files = [
        raw_dir / "project_descriptions.csv",
        raw_dir / "extended_project_description.csv",
        raw_dir / "project_metadata.csv",
        raw_dir / "projects.csv",
        raw_dir.parent / "giverep" / "raw" / "giverep_with_rootdata_descriptions_cleaned.csv",
        raw_dir.parent.parent / "giverep" / "raw" / "giverep_with_rootdata_descriptions_cleaned.csv",
    ]

    for path in candidate_files:
        if not path.exists():
            continue
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        df = df.copy()
        # Normalize header whitespace while preserving duplicate-column order.
        normalized_columns = [str(col).strip() for col in df.columns]
        df.columns = normalized_columns

        if "project_name" not in df.columns:
            continue

        description_col_names = {
            "project_description",
            "description",
            "rootdata_brief_description",
            "rootdata_full_description",
        }
        description_col_indices = [
            idx
            for idx, col_name in enumerate(normalized_columns)
            if col_name in description_col_names
        ]
        if not description_col_indices:
            continue

        for row_idx, row in df.iterrows():
            project_name = str(row["project_name"]).strip()
            if not project_name:
                continue
            project_key = _norm_key(project_name)
            for description_col_idx in description_col_indices:
                text = str(df.iat[row_idx, description_col_idx]).strip()
                if not text or text.lower() == "nan":
                    continue
                bucket = description_map.setdefault(project_key, [])
                if text not in bucket:
                    bucket.append(text)

    return description_map


def _project_descriptions(
    project_name: str,
    skill_descriptions: List[str],
    description_map: Dict[str, List[str]],
    project_name_aliases: Optional[Dict[str, str]] = None,
) -> List[str]:
    project_key = _norm_key(project_name)
    lookup_keys = [project_key]
    alias = (project_name_aliases or {}).get(project_key)
    if alias:
        lookup_keys.append(alias)

    descriptions: List[str] = []
    for key in lookup_keys:
        values = description_map.get(key, [])
        for value in values:
            if value not in descriptions:
                descriptions.append(value)
    if descriptions:
        return descriptions
    joined = "; ".join(skill_descriptions[:3])
    if joined:
        return [f"{project_name} needs a creator team aligned with these skills: {joined}."]
        return [f"{project_name} needs a creator team aligned with the required skills below."]


def _get_public_project_name(
    project_name: str,
    project_id: int,
    project_name_aliases: Optional[Dict[str, str]] = None,
) -> str:
    project_key = _norm_key(project_name)
    alias = (project_name_aliases or {}).get(project_key)
    if alias:
        return alias
    return f"P_{int(project_id)}"


def _sanitize_project_text(
    text: str,
    *,
    project_name: str,
    public_project_name: str,
    project_name_aliases: Optional[Dict[str, str]] = None,
) -> str:
    sanitized = str(text)
    replacement_keys = [project_name]

    project_key = _norm_key(project_name)
    alias = (project_name_aliases or {}).get(project_key)
    if alias:
        replacement_keys.append(alias)

    for value in sorted({key for key in replacement_keys if key}, key=len, reverse=True):
        sanitized = sanitized.replace(value, public_project_name)
    return sanitized


def _assign_yap_buckets(creators: List[Dict[str, Any]]) -> Dict[str, List[str]]:
    sorted_creators = sorted(
        creators,
        key=lambda x: (float(x["yap_score"]), str(x["creator_id"])),
    )
    n = len(sorted_creators)

    if n == 0:
        return {"low": [], "mid": [], "high": []}

    base = n // 3
    remainder = n % 3
    sizes = [
        base + (1 if remainder > 0 else 0),
        base + (1 if remainder > 1 else 0),
        base,
    ]
    bucket_names = ["low", "mid", "high"]

    groups = {bucket: [] for bucket in bucket_names}
    start = 0
    for bucket_name, size in zip(bucket_names, sizes):
        end = start + size
        for creator in sorted_creators[start:end]:
            creator["yap_bucket"] = bucket_name
            groups[bucket_name].append(creator["creator_id"])
        start = end

    return groups


def _join_creator_ids(creators: Iterable[Dict[str, Any]]) -> str:
    return " ".join(str(creator["creator_id"]) for creator in creators)


def _format_skill_descriptions(project_skills: List[str], skill_id_to_desc: Dict[str, str]) -> str:
    return "\n".join(
        f"{sid}: {skill_id_to_desc.get(str(sid).strip(), 'N/A')}" for sid in project_skills
    )


def _build_top_prompt(
    *,
    template: str,
    project_name: str,
    project_description: str,
    skill_descriptions: str,
    k: int,
) -> str:
    return template.format(
        project_name=project_name,
        project_description=project_description,
        skill_descriptions=skill_descriptions,
        k=k,
    )


def _build_relevance_prompt(
    *,
    template: str,
    project_name: str,
    project_description: str,
    skill_descriptions: str,
    k: int,
) -> str:
    return template.format(
        project_name=project_name,
        project_description=project_description,
        skill_descriptions=skill_descriptions,
        k=k,
    )


def _build_exact_range_prompt(
    *,
    template: str,
    project_name: str,
    project_description: str,
    skill_descriptions: str,
    start_rank: int,
    end_rank: int,
) -> str:
    return template.format(
        project_name=project_name,
        project_description=project_description,
        skill_descriptions=skill_descriptions,
        start_rank=start_rank,
        end_rank=end_rank,
    )


def _sort_creators_for_output(creators: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return sorted(
        creators,
        key=lambda x: (-float(x["label"]), str(x["creator_id"])),
    )


def _has_full_range(creators_by_rank: List[Dict[str, Any]], start_rank: int, end_rank: int) -> bool:
    if len(creators_by_rank) < end_rank:
        return False
    expected = list(range(start_rank, end_rank + 1))
    actual = [int(creator["rank"]) for creator in creators_by_rank[start_rank - 1 : end_rank]]
    return actual == expected


def _slice_rank_range(
    creators_by_rank: List[Dict[str, Any]],
    start_rank: int,
    end_rank: int,
) -> List[Dict[str, Any]]:
    return creators_by_rank[start_rank - 1 : end_rank]


def _build_range_target_specs(n_creators: int) -> List[Dict[str, int]]:
    specs = []
    for start_rank in range(11, 101, 10):
        if n_creators < start_rank:
            break
        end_rank = min(start_rank + 9, n_creators)
        specs.append(
            {
                "target_type": f"range_{start_rank}_{end_rank}",
                "start_rank": start_rank,
                "end_rank": end_rank,
            }
        )
    return specs


def _build_range5_target_specs(n_creators: int) -> List[Dict[str, int]]:
    specs = []
    for start_rank in range(11, 36, 5):
        end_rank = start_rank + 4
        if n_creators < end_rank:
            break
        specs.append(
            {
                "target_type": f"range_{start_rank}_{end_rank}",
                "start_rank": start_rank,
                "end_rank": end_rank,
            }
        )
    return specs


def _should_keep_short_partial_range(
    target_creators: List[Dict[str, Any]],
    seen_target_creator_ids: Optional[Set[str]],
) -> bool:
    target_creator_ids = {
        str(creator["creator_id"])
        for creator in target_creators
    }
    if not target_creator_ids:
        return False
    if seen_target_creator_ids is None:
        return True
    return any(
        creator_id not in seen_target_creator_ids
        for creator_id in target_creator_ids
    )


def _load_project_context(
    *,
    raw_dir: str,
    skills_csv: str,
    creators_csv: str,
    gpt5_skills_csv: str,
    creator_details_csv: str,
) -> Dict[str, Any]:
    raw_dir_path = Path(raw_dir)
    skills_df = pd.read_csv(skills_csv)
    creators_df = pd.read_csv(creators_csv)
    skill_desc_df = pd.read_csv(gpt5_skills_csv)
    creator_rank_df = pd.read_csv(creator_details_csv)

    yaps_map = load_creator_yaps(raw_dir_path)
    description_map = _load_project_description_variants(raw_dir_path)
    project_name_aliases = _load_project_name_aliases(raw_dir_path)

    if "id" not in skill_desc_df.columns or "one_sentence_definition" not in skill_desc_df.columns:
        raise KeyError("gpt5_skills_csv must contain columns: id, one_sentence_definition")

    skill_id_to_desc = {
        str(row["id"]).strip(): str(row["one_sentence_definition"]).strip()
        for _, row in skill_desc_df.iterrows()
    }

    if not {"project_id", "project_name", "assigned_skill_ids"}.issubset(set(skills_df.columns)):
        raise KeyError("skills_csv must contain columns: project_id, project_name, assigned_skill_ids")
    if "project_name" not in creators_df.columns or "creators" not in creators_df.columns:
        raise KeyError("creators_csv must contain columns: project_name, creators")

    skills_map: Dict[str, List[str]] = {}
    project_id_map: Dict[str, int] = {}
    canonical_name: Dict[str, str] = {}
    for _, row in skills_df.iterrows():
        pname = str(row["project_name"]).strip()
        if not pname:
            continue
        key = _norm_key(pname)
        canonical_name[key] = pname
        skills_map[key] = _safe_list_parse(row["assigned_skill_ids"])
        project_id_map[key] = int(row["project_id"])

    creators_map = {}
    for _, row in creators_df.iterrows():
        pname = str(row["project_name"]).strip()
        if not pname:
            continue
        creators_map[_norm_key(pname)] = _safe_list_parse(row["creators"])

    need_cols = {"creator_code", "project_name", "Rank"}
    if not need_cols.issubset(set(creator_rank_df.columns)):
        raise KeyError(
            f"creator_details_csv must contain columns: {sorted(need_cols)}"
        )

    creator_rank_df = creator_rank_df.copy()
    creator_rank_df["project_name_norm"] = creator_rank_df["project_name"].astype(str).map(_norm_key)
    creator_rank_df["creator_code"] = creator_rank_df["creator_code"].astype(str)
    creator_rank_df["Rank"] = pd.to_numeric(creator_rank_df["Rank"], errors="coerce")
    creator_rank_df = creator_rank_df.dropna(subset=["Rank"])

    return {
        "skills_df": skills_df,
        "skill_id_to_desc": skill_id_to_desc,
        "skills_map": skills_map,
        "project_id_map": project_id_map,
        "canonical_name": canonical_name,
        "creators_map": creators_map,
        "creator_rank_df": creator_rank_df,
        "yaps_map": yaps_map,
        "description_map": description_map,
        "project_name_aliases": project_name_aliases,
    }


def _build_project_record_base(
    *,
    project_key: str,
    context: Dict[str, Any],
    max_candidates: int,
) -> Optional[Dict[str, Any]]:
    skills_map = context["skills_map"]
    creators_map = context["creators_map"]
    canonical_name = context["canonical_name"]
    project_id_map = context["project_id_map"]
    skill_id_to_desc = context["skill_id_to_desc"]
    creator_rank_df = context["creator_rank_df"]
    yaps_map = context["yaps_map"]
    description_map = context["description_map"]
    project_name_aliases = context["project_name_aliases"]

    if project_key not in skills_map or project_key not in creators_map:
        return None

    project_name = canonical_name[project_key]
    project_id = project_id_map[project_key]
    public_project_name = _get_public_project_name(
        project_name,
        project_id,
        project_name_aliases,
    )
    project_skills = skills_map[project_key]
    skill_descriptions_text = _format_skill_descriptions(project_skills, skill_id_to_desc)
    project_descriptions = _project_descriptions(
        project_name,
        skill_descriptions_text.splitlines(),
        description_map,
        project_name_aliases,
    )
    project_descriptions = [
        _sanitize_project_text(
            description,
            project_name=project_name,
            public_project_name=public_project_name,
            project_name_aliases=project_name_aliases,
        )
        for description in project_descriptions
    ]

    raw_project_creator_ids = [str(creator_id) for creator_id in creators_map[project_key]]
    seen_creator_ids = set()
    full_project_creator_ids = []
    for creator_id in raw_project_creator_ids:
        if creator_id in seen_creator_ids:
            continue
        seen_creator_ids.add(creator_id)
        full_project_creator_ids.append(creator_id)
    participant_creators = set(full_project_creator_ids)

    if len(full_project_creator_ids) > max_candidates:
        raise ValueError(
            f"Project '{public_project_name}' has {len(full_project_creator_ids)} creators in creators.csv, "
            f"which exceeds max_candidates={max_candidates}. Increase --max_candidates."
        )

    df_pos = creator_rank_df[
        (creator_rank_df["project_name_norm"] == project_key)
        & (creator_rank_df["creator_code"].isin(participant_creators))
    ].copy()

    if df_pos.empty:
        log.warning(
            "[Skip] No positives found for project='%s' (pkey='%s').",
            public_project_name,
            project_key,
        )
        return None

    df_pos = df_pos.sort_values(["Rank", "creator_code"], ascending=[True, True]).reset_index(drop=True)
    df_pos["new_rank"] = range(1, len(df_pos) + 1)
    n_pos = len(df_pos)

    creators_by_rank = []
    for _, row in df_pos.iterrows():
        dense_rank = int(row["new_rank"])
        corrected_rank = n_pos - dense_rank + 1
        creators_by_rank.append(
            {
                "creator_id": str(row["creator_code"]),
                "rank": dense_rank,
                "label": float(t_from_rank(dense_rank, n_pos)),
                "corrected_rank": corrected_rank,
                "yap_score": float(yaps_map.get(str(row["creator_code"]), 0.0)),
            }
        )

    ranked_creator_map = {
        str(creator["creator_id"]): creator
        for creator in creators_by_rank
    }
    creators_for_output = []
    for creator_id in full_project_creator_ids:
        ranked_creator = ranked_creator_map.get(str(creator_id))
        creators_for_output.append(
            {
                "creator_id": str(creator_id),
                "rank": None if ranked_creator is None else int(ranked_creator["corrected_rank"]),
                "label": 0.0 if ranked_creator is None else float(ranked_creator["label"]),
                "yap_score": float(yaps_map.get(str(creator_id), 0.0)),
            }
        )

    yap_bucket_groups = _assign_yap_buckets(creators_for_output)
    creators_for_output = _sort_creators_for_output(creators_for_output)

    return {
        "project_id": project_id,
        "project_name": public_project_name,
        "project_descriptions": project_descriptions,
        "skill_descriptions": skill_descriptions_text,
        "creators_by_rank": creators_by_rank,
        "creators_for_output": creators_for_output,
        "yap_bucket_groups": yap_bucket_groups,
    }


def _build_training_records(
    project_base: Dict[str, Any],
    split: str,
    templates: Dict[str, str],
    seen_target_creator_ids: Optional[Set[str]] = None,
) -> List[Dict[str, Any]]:
    records = []
    seen_record_keys = set()
    creators_by_rank = project_base["creators_by_rank"]
    creators_for_output = project_base["creators_for_output"]
    record_project_id = project_base["project_id"]
    project_descriptions = project_base["project_descriptions"]

    shared = {
        "creators": creators_for_output,
        "split": split,
    }

    def add_record(record: Dict[str, Any]) -> None:
        dedupe_key = (
            record_project_id,
            record["prompt"],
            record["target_text"],
            record["target_type"],
        )
        if dedupe_key in seen_record_keys:
            return
        seen_record_keys.add(dedupe_key)
        records.append(record)
        if seen_target_creator_ids is not None:
            seen_target_creator_ids.update(record["target_text"].split())

    for project_description in project_descriptions:
        for spec in TOP_TARGET_SPECS:
            if not _has_full_range(creators_by_rank, spec["start_rank"], spec["end_rank"]):
                continue
            target_creators = _slice_rank_range(
                creators_by_rank,
                spec["start_rank"],
                spec["end_rank"],
            )
            target_text = _join_creator_ids(target_creators)

            add_record(
                {
                    "prompt": _build_top_prompt(
                        template=templates["top"],
                        project_name=project_base["project_name"],
                        project_description=project_description,
                        skill_descriptions=project_base["skill_descriptions"],
                        k=spec["k"],
                    ),
                    "target_text": target_text,
                    "target_type": spec["target_type"],
                    **shared,
                }
            )
            add_record(
                {
                    "prompt": _build_relevance_prompt(
                        template=templates["relevance"],
                        project_name=project_base["project_name"],
                        project_description=project_description,
                        skill_descriptions=project_base["skill_descriptions"],
                        k=spec["k"],
                    ),
                    "target_text": target_text,
                    "target_type": spec["target_type"],
                    **shared,
                }
            )

        for spec in _build_range_target_specs(len(creators_by_rank)):
            target_creators = _slice_rank_range(
                creators_by_rank,
                spec["start_rank"],
                spec["end_rank"],
            )
            target_len = len(target_creators)
            if target_len == 0:
                continue
            if target_len != (spec["end_rank"] - spec["start_rank"] + 1):
                continue
            if target_len <= 4 and not _should_keep_short_partial_range(
                target_creators,
                seen_target_creator_ids,
            ):
                continue
            add_record(
                {
                    "prompt": _build_exact_range_prompt(
                        template=templates["exact_range"],
                        project_name=project_base["project_name"],
                        project_description=project_description,
                        skill_descriptions=project_base["skill_descriptions"],
                        start_rank=spec["start_rank"],
                        end_rank=spec["end_rank"],
                    ),
                    "target_text": _join_creator_ids(target_creators),
                    "target_type": spec["target_type"],
                    **shared,
                }
            )

        for spec in _build_range5_target_specs(len(creators_by_rank)):
            if not _has_full_range(creators_by_rank, spec["start_rank"], spec["end_rank"]):
                continue
            target_creators = _slice_rank_range(
                creators_by_rank,
                spec["start_rank"],
                spec["end_rank"],
            )
            add_record(
                {
                    "prompt": _build_exact_range_prompt(
                        template=templates["exact_range"],
                        project_name=project_base["project_name"],
                        project_description=project_description,
                        skill_descriptions=project_base["skill_descriptions"],
                        start_rank=spec["start_rank"],
                        end_rank=spec["end_rank"],
                    ),
                    "target_text": _join_creator_ids(target_creators),
                    "target_type": spec["target_type"],
                    **shared,
                }
            )

    return records


def _build_test_records(
    project_base: Dict[str, Any],
    split: str,
    templates: Dict[str, str],
) -> List[Dict[str, Any]]:
    records = []
    creators_by_rank = project_base["creators_by_rank"]
    creators_for_output = project_base["creators_for_output"]
    project_descriptions = project_base["project_descriptions"]

    shared = {
        "creators": creators_for_output,
        "split": split,
    }

    for project_description in project_descriptions:
        for spec in TEST_TARGET_SPECS:
            if not _has_full_range(creators_by_rank, 1, spec["k"]):
                continue
            target_creators = _slice_rank_range(creators_by_rank, 1, spec["k"])
            records.append(
                {
                    "prompt": _build_relevance_prompt(
                        template=templates["relevance"],
                        project_name=project_base["project_name"],
                        project_description=project_description,
                        skill_descriptions=project_base["skill_descriptions"],
                        k=spec["k"],
                    ),
                    "target_text": _join_creator_ids(target_creators),
                    "target_type": spec["target_type"],
                    **shared,
                }
            )

    return records


def build_team_formation_jsonl_for_projects(
    *,
    projects: List[str],
    raw_dir: str,
    skills_csv: str,
    creators_csv: str,
    gpt5_skills_csv: str,
    creator_details_csv: str,
    output_jsonl: str,
    max_candidates: int = 128,
    split: str,
) -> int:
    templates = {
        "top": _load_prompt_template(TOP_PROMPT_TEMPLATE_PATH),
        "relevance": _load_prompt_template(RELEVANCE_PROMPT_TEMPLATE_PATH),
        "exact_range": _load_prompt_template(EXACT_RANGE_PROMPT_TEMPLATE_PATH),
    }
    context = _load_project_context(
        raw_dir=raw_dir,
        skills_csv=skills_csv,
        creators_csv=creators_csv,
        gpt5_skills_csv=gpt5_skills_csv,
        creator_details_csv=creator_details_csv,
    )

    written = 0
    skipped = 0
    seen_target_creator_ids: Optional[Set[str]] = set() if split != "test" else None
    os.makedirs(os.path.dirname(output_jsonl) or ".", exist_ok=True)

    with open(output_jsonl, "w", encoding="utf-8") as f:
        for project in projects:
            project_key = _norm_key(project)
            project_base = _build_project_record_base(
                project_key=project_key,
                context=context,
                max_candidates=max_candidates,
            )
            if project_base is None:
                skipped += 1
                continue

            if split == "test":
                records = _build_test_records(project_base, split, templates)
            else:
                records = _build_training_records(
                    project_base,
                    split,
                    templates,
                    seen_target_creator_ids=seen_target_creator_ids,
                )

            if not records:
                skipped += 1
                continue

            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                written += 1

    print(f"[Team Formation JSONL] split={split} wrote={written}, skipped={skipped}, path={output_jsonl}")
    return written


def _combine_jsonl_files(input_paths: List[str], output_jsonl: str) -> int:
    os.makedirs(os.path.dirname(output_jsonl) or ".", exist_ok=True)
    written = 0
    with open(output_jsonl, "w", encoding="utf-8") as fout:
        for input_path in input_paths:
            if not os.path.exists(input_path):
                continue
            with open(input_path, "r", encoding="utf-8") as fin:
                for line in fin:
                    if not line.strip():
                        continue
                    fout.write(line)
                    written += 1
    print(f"[Team Formation JSONL] split=combined wrote={written}, path={output_jsonl}")
    return written


def build_one_trial_one_fold(
    *,
    splits_pkl: str,
    indexes_pkl: str,
    teamsvecs_pkl: Optional[str],
    trial_id: int,
    fold_id: int,
    out_train_jsonl: str,
    out_valid_jsonl: str,
    out_test_jsonl: str,
    out_combined_jsonl: str,
    raw_dir: str,
    skills_csv: str,
    creators_csv: str,
    gpt5_skills_csv: str,
    creator_details_csv: str,
    max_candidates: int = 128,
) -> Dict[str, int]:
    bundle = _load_splits_bundle(splits_pkl)
    trial = bundle["trials"][int(trial_id)]
    fold = trial["folds"][int(fold_id)]

    indexes = _load_indexes(indexes_pkl)

    if teamsvecs_pkl:
        shape = _maybe_load_teamsvecs_shape(teamsvecs_pkl)
        if shape is not None:
            n_teams, _ = shape
            i2t_len = len(indexes["i2t"]) if not isinstance(indexes["i2t"], dict) else len(indexes["i2t"])
            if i2t_len != n_teams:
                log.warning(
                    "[Sanity] indexes.i2t length (%s) != teamsvecs n_teams (%s).",
                    i2t_len,
                    n_teams,
                )

    skills_df = pd.read_csv(skills_csv)

    train_idx = list(map(int, fold["train"]))
    valid_idx = list(map(int, fold["valid"]))
    test_idx = []

    if "test" in fold:
        test_idx = list(map(int, fold["test"]))
    elif "test" in trial:
        test_idx = list(map(int, trial["test"]))

    train_projects = _row_indices_to_project_names(train_idx, indexes, skills_df)
    valid_projects = _row_indices_to_project_names(valid_idx, indexes, skills_df)
    test_projects = _row_indices_to_project_names(test_idx, indexes, skills_df)

    print(
        f"[Split] trial={trial_id} fold={fold_id} "
        f"train={len(train_projects)} valid={len(valid_projects)} test={len(test_projects)}"
    )

    train_written = build_team_formation_jsonl_for_projects(
        projects=train_projects,
        raw_dir=raw_dir,
        skills_csv=skills_csv,
        creators_csv=creators_csv,
        gpt5_skills_csv=gpt5_skills_csv,
        creator_details_csv=creator_details_csv,
        output_jsonl=out_train_jsonl,
        max_candidates=max_candidates,
        split="train",
    )

    valid_written = build_team_formation_jsonl_for_projects(
        projects=valid_projects,
        raw_dir=raw_dir,
        skills_csv=skills_csv,
        creators_csv=creators_csv,
        gpt5_skills_csv=gpt5_skills_csv,
        creator_details_csv=creator_details_csv,
        output_jsonl=out_valid_jsonl,
        max_candidates=max_candidates,
        split="valid",
    )

    test_written = build_team_formation_jsonl_for_projects(
        projects=test_projects,
        raw_dir=raw_dir,
        skills_csv=skills_csv,
        creators_csv=creators_csv,
        gpt5_skills_csv=gpt5_skills_csv,
        creator_details_csv=creator_details_csv,
        output_jsonl=out_test_jsonl,
        max_candidates=max_candidates,
        split="test",
    )

    combined_written = _combine_jsonl_files(
        [out_train_jsonl, out_valid_jsonl, out_test_jsonl],
        out_combined_jsonl,
    )
    total_written = train_written + valid_written + test_written
    print(
        "[Team Formation JSONL] summary "
        f"train={train_written} valid={valid_written} test={test_written} "
        f"total={total_written} combined={combined_written}"
    )
    return {
        "train": train_written,
        "valid": valid_written,
        "test": test_written,
        "total": total_written,
        "combined": combined_written,
    }


def parse_args():
    repo_root = Path(__file__).resolve().parents[2]
    default_data_root = repo_root / "data" / "kaito" / "raw"
    default_out_dir = repo_root / "data" / "kaito"
    parser = argparse.ArgumentParser(
        description="Build compact creator-generation JSONL (train + valid + test) for one trial & one fold"
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default=str(default_data_root),
        help="Path to raw data directory (contains skills.csv, creators.csv, splits.pkl, etc.)",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=str(default_out_dir),
        help="Output directory for processed JSONL files",
    )
    parser.add_argument("--trial_id", type=int, default=0, help="Trial index")
    parser.add_argument("--fold_id", type=int, default=0, help="Fold index")
    parser.add_argument(
        "--max_candidates",
        type=int,
        default=128,
        help="Maximum number of candidate creators per project prompt",
    )
    return parser.parse_args()


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    )
    args = parse_args()

    data_root = Path(args.data_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    splits_pkl = data_root / "splits.t5.r0.85.pkl"
    indexes_pkl = data_root / "indexes.pkl"
    teamsvecs_pkl = data_root / "teamsvecs.pkl"
    skills_csv = data_root / "skills.csv"
    creators_csv = data_root / "creators.csv"
    gpt5_csv = data_root / "gpt5_skills.csv"
    details_csv = data_root / "creator_details.csv"

    preprocess(str(data_root))

    counts = build_one_trial_one_fold(
        splits_pkl=str(splits_pkl),
        indexes_pkl=str(indexes_pkl),
        teamsvecs_pkl=str(teamsvecs_pkl),
        trial_id=args.trial_id,
        fold_id=args.fold_id,
        out_train_jsonl=str(out_dir / "train.jsonl"),
        out_valid_jsonl=str(out_dir / "valid.jsonl"),
        out_test_jsonl=str(out_dir / "test.jsonl"),
        out_combined_jsonl=str(out_dir / "combined.jsonl"),
        raw_dir=str(data_root),
        skills_csv=str(skills_csv),
        creators_csv=str(creators_csv),
        gpt5_skills_csv=str(gpt5_csv),
        creator_details_csv=str(details_csv),
        max_candidates=args.max_candidates,
    )
    print(
        "[Done] "
        f"train={counts['train']} valid={counts['valid']} "
        f"test={counts['test']} total={counts['total']} combined={counts['combined']}"
    )


if __name__ == "__main__":
    main()
