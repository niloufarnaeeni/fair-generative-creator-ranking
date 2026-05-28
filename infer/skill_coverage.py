# skill_coverage.py
from __future__ import annotations

import re
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import scipy.sparse as sp


_QID_META_RE = re.compile(r"\|t(\d+)\|f(\d+)\|(\w+)\|")
_CID_RE = re.compile(r"^C(\d+)$", re.IGNORECASE)


def parse_qid_meta(qid: str) -> Tuple[int, int, str]:
    """
    qid is built like:
      "{project}|t{trial_id}|f{fold_id}|{split}|i{idx:04d}"
    """
    m = _QID_META_RE.search(qid)
    if not m:
        return 0, 0, "unknown"
    return int(m.group(1)), int(m.group(2)), m.group(3)


def get_skipteams_from_splits(
    splits_bundle: dict,
    trial_id: int,
    fold_id: int,
    split_name: str,
) -> Optional[List[int]]:
    """
    To avoid leakage: when evaluating on valid/test, exclude those team rows
    from the co-occurrence matrix build (same logic as Team.gen_skill_coverage).
    """
    if split_name not in {"valid", "test"}:
        return None

    trial = splits_bundle["trials"][int(trial_id)]
    fold = trial["folds"][int(fold_id)]

    if split_name == "valid":
        return list(map(int, fold["valid"]))

    # split_name == "test"
    if "test" in fold:
        return list(map(int, fold["test"]))
    if "test" in trial:  # legacy
        return list(map(int, trial["test"]))
    return None


def gen_member_skill_cooccurrence(
    teamsvecs: dict,
    cache_path: Path,
    skipteams: Optional[List[int]] = None,
) -> sp.csr_matrix:
    """
    Equivalent to Team.gen_skill_coverage:
      member_skill_co = member^T @ skill
    Shape: (n_members, n_skills)
    """
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    if cache_path.exists():
        with open(cache_path, "rb") as f:
            mat = pickle.load(f)

        exp_shape = (teamsvecs["member"].shape[1], teamsvecs["skill"].shape[1])
        if getattr(mat, "shape", None) != exp_shape:
            raise ValueError(f"Cached skillcoverage wrong shape {mat.shape}, expected {exp_shape}")
        return mat

    member = teamsvecs["member"].copy()
    skill = teamsvecs["skill"].copy()

    # teamsvecs are usually LIL; clearing rows is cheap in LIL
    if skipteams:
        for i in skipteams:
            member.rows[i] = []
            member.data[i] = []
            skill.rows[i] = []
            skill.data[i] = []

    mat = sp.csr_matrix(member.transpose().tocsr() @ skill.tocsr())

    with open(cache_path, "wb") as f:
        pickle.dump(mat, f)

    return mat


def build_Cid_to_member_index(indexes: dict) -> Dict[str, int]:
    """
    Your evaluation docids are like: "C169".
    Your indexes member IDs look like: "0_c169" (example pattern from your i2c).

    This builds a robust mapping:
      "C169" / "c169" -> member_index
      "0_c169" -> member_index   (if needed)
      "c169" -> member_index     (suffix-only)
    """
    out: Dict[str, int] = {}

    # c2i: internal_id -> member_index
    c2i = indexes["c2i"]

    for internal_id, midx in c2i.items():
        internal_id = str(internal_id)      # e.g., "0_c3"
        midx = int(midx)

        out[internal_id] = midx

        parts = internal_id.split("_", 1)
        if len(parts) == 2:
            suffix = parts[1]               # e.g., "c3"
            out[suffix.lower()] = midx      # "c3"
            out[suffix.upper()] = midx      # "C3"

    # also allow mapping from "C3" -> "c3" if not already present
    # (usually already covered by suffix.upper())
    return out


def extract_required_skill_indices_from_query(query_text: str, s2i: dict) -> List[int]:
    """
    You embed skill IDs like 'S23' in the query (skill_ids=...).
    We scan for tokens that exist in s2i keys.
    """
    q = str(query_text)
    req = []
    for token, idx in s2i.items():
        token = str(token)
        if re.search(rf"(?<!\w){re.escape(token)}(?!\w)", q):
            req.append(int(idx))
    return req


def compute_skill_coverage_at_k(
    run: Dict[str, Dict[str, float]],
    qid_to_query: Dict[str, str],
    indexes: dict,
    member_skill_co: sp.csr_matrix,
    ks: Tuple[int, ...] = (2, 5, 10),
) -> Dict[str, Dict[str, float]]:
    """
    SkillCoverage@k = |RequiredSkills ∩ UnionSkills(TopK creators)| / |RequiredSkills|

    - RequiredSkills extracted from query using s2i tokens (e.g., S23).
    - Creator skills come from member_skill_co (member x skill).
    - Docids in run are like "C169"; we map them to member indices using indexes.pkl.
    """
    s2i = indexes["s2i"]
    docid_to_member_idx = build_Cid_to_member_index(indexes)

    # boolean matrix for fast union (indices list)
    member_skill_bool = (member_skill_co > 0).tocsr()

    out: Dict[str, Dict[str, float]] = {}

    for qid, doc_scores in run.items():
        out[qid] = {}
        query_text = qid_to_query.get(qid, "")

        req = set(extract_required_skill_indices_from_query(query_text, s2i))
        if not req:
            for k in ks:
                out[qid][f"skill_coverage_{k}"] = 0.0
            continue

        ranked = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        denom = len(req)

        for k in ks:
            union_skills = set()

            for docid, _ in ranked[:k]:
                docid = str(docid)

                midx = docid_to_member_idx.get(docid)
                if midx is None:
                    midx = docid_to_member_idx.get(docid.lower())

                # fallback: if docid is exactly "C169", try turning into "c169"
                if midx is None:
                    m = _CID_RE.match(docid)
                    if m:
                        midx = docid_to_member_idx.get(f"c{m.group(1)}")

                if midx is None:
                    continue

                union_skills.update(member_skill_bool[midx].indices)

            out[qid][f"skill_coverage_{k}"] = len(req.intersection(union_skills)) / denom

    return out
