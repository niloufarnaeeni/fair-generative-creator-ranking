from __future__ import annotations


import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parent))
    from eval_metrics import (  # type: ignore
        add_group_exposure_metrics_to_per_query,
        add_yaps_metrics_to_agg,
        aggregate_metrics,
        append_agg_metrics_csv,
        compute_within_group_concentration_metrics,
        evaluate_binary_and_graded,
        load_run_config_or_default,
        save_metrics_csv,
    )
else:
    from .eval_metrics import (
        add_group_exposure_metrics_to_per_query,
        add_yaps_metrics_to_agg,
        aggregate_metrics,
        append_agg_metrics_csv,
        compute_within_group_concentration_metrics,
        evaluate_binary_and_graded,
        load_run_config_or_default,
        save_metrics_csv,
    )


RERANKER_TRAIN_DIR = Path(__file__).resolve().parents[1] / "train" / "reranker"
if str(RERANKER_TRAIN_DIR) not in sys.path:
    sys.path.append(str(RERANKER_TRAIN_DIR))

from model_llm import LLMDecoder  # noqa: E402
GROUP_NAME_TO_ID = {"low": 0, "mid": 1, "high": 2}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate team-formation creator generation against creator-level rank/label annotations."
    )
    parser.add_argument("--jsonl", required=True, help="Path to team-formation test JSONL")
    parser.add_argument("--model", required=True, help="Local path to saved LLM decoder checkpoint")
    parser.add_argument("--output_dir", default="eval_outputs", help="Output directory")
    parser.add_argument(
        "--results_csv_name",
        default="results.csv",
        help="Filename for the shared aggregated results CSV under the dataset eval directory.",
    )
    parser.add_argument("--ks", default="10,20", help="Comma-separated cutoffs, e.g. 10,20")
    parser.add_argument(
        "--relevance_gt",
        type=float,
        default=0.0,
        help="Binary relevance threshold: label > relevance_gt is relevant",
    )
    parser.add_argument("--max_new_tokens", type=int, default=32, help="Max creator tokens to generate")
    parser.add_argument("--batch_size", type=int, default=1, help="Generation batch size")
    parser.add_argument("--device", default=None, help="Optional device override, e.g. cuda:0 or cpu")
    parser.add_argument(
        "--compare_with_baseline",
        action="store_true",
        default=False,
        help="Append a baseline comparison CSV beside the aggregate results CSV.",
    )
    parser.add_argument("--verbose", type=int, default=1)
    return parser.parse_args()


def _load_llm_decoder(
    model_path: str,
    device: Optional[str] = None,
    max_len: Optional[int] = None,
    max_creator_candidates: Optional[int] = None,
) -> LLMDecoder:
    decoder = LLMDecoder.from_pretrained(
        model_path,
        task_type="team_formation",
        max_len=max_len or 2048,
        max_creator_candidates=max_creator_candidates or 64,
    )
    decoder.model.eval()

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    decoder.model.to(device)
    return decoder


def _dedupe_preserve_order(values: Sequence[str]) -> List[str]:
    seen = set()
    deduped = []
    for value in values:
        value = str(value).strip()
        if not value or value in seen:
            continue
        seen.add(value)
        deduped.append(value)
    return deduped


def match_predicted_relevance(
    predicted_creator_ids: Sequence[str],
    label_map: Optional[Dict[str, float]] = None,
) -> List[float]:
    label_map = label_map or {}
    return [label_map.get(str(creator_id), 0.0) for creator_id in predicted_creator_ids]


def extract_eval_targets_from_team_formation_row(
    row: Dict,
    *,
    qid: str,
) -> Tuple[List[str], List[float], Dict[str, int]]:
    creators = row.get("creators", []) or []
    if not creators:
        raise ValueError(f"Row for qid={qid} is missing creators.")

    candidate_creator_ids: List[str] = []
    candidate_labels: List[float] = []
    candidate_gains: Dict[str, int] = {}

    for creator in creators:
        creator_id = creator.get("creator_id")
        if creator_id is None:
            continue

        creator_id = str(creator_id)
        candidate_creator_ids.append(creator_id)
        candidate_labels.append(float(creator.get("label", 0.0)))

        rank = creator.get("rank")
        if rank is None:
            continue
        rank_int = int(rank)
        if rank_int > 0:
            candidate_gains[creator_id] = rank_int

    if not candidate_creator_ids:
        raise ValueError(f"Row for qid={qid} is missing valid creator_id entries.")

    positives_sorted = sorted(
        zip(candidate_creator_ids, candidate_labels),
        key=lambda x: (-float(x[1]), str(x[0])),
    )
    n_candidates = len(positives_sorted)
    candidate_gains = {
        str(creator_id): n_candidates - rank_idx + 1
        for rank_idx, (creator_id, _) in enumerate(positives_sorted, start=1)
    }

    return candidate_creator_ids, candidate_labels, candidate_gains


def load_creator_yaps_from_team_formation_jsonl(jsonl_path: Path) -> Dict[str, float]:
    creator_yaps: Dict[str, float] = {}

    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            for creator in row.get("creators", []) or []:
                creator_id = creator.get("creator_id")
                if creator_id is None:
                    continue
                creator_yaps[str(creator_id)] = float(creator.get("yap_score", 0.0))

    return creator_yaps


def _derive_group_map_from_creators(creators: Sequence[Dict]) -> Dict[str, int]:
    explicit_group_map = {}
    explicit_buckets = [creator.get("yap_bucket") for creator in creators]
    if explicit_buckets and all(bucket in GROUP_NAME_TO_ID for bucket in explicit_buckets):
        for creator in creators:
            creator_id = creator.get("creator_id")
            bucket = creator.get("yap_bucket")
            if creator_id is None or bucket not in GROUP_NAME_TO_ID:
                continue
            explicit_group_map[str(creator_id)] = GROUP_NAME_TO_ID[str(bucket)]
        return explicit_group_map

    scored_creators = sorted(
        [
            (
                float(creator.get("yap_score", 0.0)),
                str(creator.get("creator_id")),
            )
            for creator in creators
            if creator.get("creator_id") is not None
        ],
        key=lambda item: (item[0], item[1]),
    )
    n = len(scored_creators)
    if n == 0:
        return {}

    base = n // 3
    remainder = n % 3
    sizes = [
        base + (1 if remainder > 0 else 0),
        base + (1 if remainder > 1 else 0),
        base,
    ]

    group_map: Dict[str, int] = {}
    start = 0
    for group_id, size in enumerate(sizes):
        end = start + size
        for _, creator_id in scored_creators[start:end]:
            group_map[creator_id] = group_id
        start = end
    return group_map


def extract_creator_group_map_from_team_formation_row(
    row: Dict,
    *,
    qid: str,
) -> Dict[str, int]:
    def _group_map_from_bucket_groups(creator_ids: Sequence[str]) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for bucket_name, bucket_creator_ids in (row.get("yap_bucket_groups") or {}).items():
            if bucket_name not in GROUP_NAME_TO_ID:
                continue
            for creator_id in bucket_creator_ids or []:
                creator_id = str(creator_id)
                if creator_id in creator_ids:
                    out[creator_id] = GROUP_NAME_TO_ID[bucket_name]
        return out

    def _group_map_from_aligned_group_ids(creator_ids: Sequence[str]) -> Dict[str, int]:
        raw_group_ids = row.get("project_group_ids") or row.get("eval_group_ids") or []
        if len(raw_group_ids) != len(creator_ids):
            return {}
        out: Dict[str, int] = {}
        for creator_id, raw_group_id in zip(creator_ids, raw_group_ids):
            try:
                group_id = int(raw_group_id)
            except (TypeError, ValueError):
                return {}
            if group_id not in GROUP_NAME_TO_ID.values():
                return {}
            out[str(creator_id)] = group_id
        return out

    creators = row.get("creators", []) or []
    if not creators:
        raise ValueError(f"Row for qid={qid} is missing creators.")

    group_map = _derive_group_map_from_creators(creators)
    if group_map:
        return group_map

    fallback = {}
    for creator in creators:
        creator_id = creator.get("creator_id")
        if creator_id is None:
            continue
        fallback[str(creator_id)] = GROUP_NAME_TO_ID["low"]
    if not fallback:
        raise ValueError(f"Row for qid={qid} is missing valid creator_id entries for group mapping.")
    return fallback


def _generate_creator_ids_with_optional_project_constraints(
    *,
    decoder,
    prompts: Sequence[str],
    max_new_tokens: int,
    project_creator_ids: Sequence[Sequence[str]],
):
    try:
        return decoder.generate_creator_ids(
            prompts,
            max_new_tokens=max_new_tokens,
            project_creator_ids=project_creator_ids,
        )
    except TypeError as exc:
        if "project_creator_ids" not in str(exc):
            raise
        return decoder.generate_creator_ids(
            prompts,
            max_new_tokens=max_new_tokens,
        )


def build_qrels_and_run_from_team_formation_jsonl(
    *,
    jsonl_path: Path,
    decoder: LLMDecoder,
    max_new_tokens: int,
    batch_size: int,
    relevance_gt: float = 0.0,
):
    qrels_binary: Dict[str, Dict[str, int]] = {}
    qrels_graded: Dict[str, Dict[str, int]] = {}
    run: Dict[str, Dict[str, float]] = {}
    qid_to_query: Dict[str, str] = {}
    qid_to_labels: Dict[str, Dict[str, float]] = {}
    qid_to_group_ids: Dict[str, Dict[str, int]] = {}
    prediction_rows = []

    rows = []
    with open(jsonl_path, encoding="utf-8") as f:
        for idx, line in enumerate(f):
            rows.append((idx, json.loads(line)))

    for start in range(0, len(rows), batch_size):
        chunk = rows[start : start + batch_size]
        prompts = [row["prompt"] for _, row in chunk]
        project_creator_ids = [
            [str(creator["creator_id"]) for creator in row.get("creators", []) or []]
            for _, row in chunk
        ]
        predictions = _generate_creator_ids_with_optional_project_constraints(
            decoder=decoder,
            prompts=prompts,
            max_new_tokens=max_new_tokens,
            project_creator_ids=project_creator_ids,
        )

        for (idx, row), predicted_creator_ids in zip(chunk, predictions):
            qid = f"{row.get('target_type', 'unknown')}|i{idx:04d}"
            candidate_creator_ids, candidate_labels, candidate_gains = extract_eval_targets_from_team_formation_row(
                row,
                qid=qid,
            )
            qid_to_group_ids[qid] = extract_creator_group_map_from_team_formation_row(
                row,
                qid=qid,
            )

            predicted_creator_ids = _dedupe_preserve_order(predicted_creator_ids)
            label_map = {
                str(creator_id): float(label)
                for creator_id, label in zip(candidate_creator_ids, candidate_labels)
            }
            matched_relevances = match_predicted_relevance(
                predicted_creator_ids=predicted_creator_ids,
                label_map=label_map,
            )

            qrels_graded[qid] = {
                str(creator_id): int(gain)
                for creator_id, gain in candidate_gains.items()
            }
            qrels_binary[qid] = {
                str(creator_id): int(float(label) > relevance_gt)
                for creator_id, label in zip(candidate_creator_ids, candidate_labels)
                if float(label) > relevance_gt
            }
            qid_to_query[qid] = str(row.get("prompt", ""))
            qid_to_labels[qid] = dict(label_map)

            # Ranking metrics are computed from predicted order; higher score means earlier rank.
            # Creators absent from the candidate creator set are treated as relevance 0 by omission from qrels.
            run[qid] = {
                str(creator_id): float(len(predicted_creator_ids) - rank_idx)
                for rank_idx, creator_id in enumerate(predicted_creator_ids)
            }

            prediction_rows.append(
                {
                    "sample_id": qid,
                    "target_type": row.get("target_type"),
                    "predicted_creator_ids": predicted_creator_ids,
                    "predicted_relevances": matched_relevances,
                    "candidate_creator_ids": candidate_creator_ids,
                    "candidate_labels": candidate_labels,
                    "candidate_gains": [
                        int(candidate_gains.get(str(creator_id), 0))
                        for creator_id in candidate_creator_ids
                    ],
                }
            )

    return (
        qrels_binary,
        qrels_graded,
        run,
        qid_to_query,
        qid_to_labels,
        qid_to_group_ids,
        prediction_rows,
    )


def main():
    args = parse_args()
    ks = tuple(int(x.strip()) for x in args.ks.split(",") if x.strip())
    jsonl_path = Path(args.jsonl).resolve()
    run_config = load_run_config_or_default(args.model)

    run_output_dir = Path(args.output_dir).resolve()
    per_run_eval_dir = run_output_dir / "eval"
    per_run_eval_dir.mkdir(parents=True, exist_ok=True)
    shared_eval_dir = (
        run_output_dir.parent.parent / "eval"
        if run_output_dir.parent.name == "run"
        else run_output_dir.parent / "eval"
    )
    shared_eval_dir.mkdir(parents=True, exist_ok=True)

    decoder = _load_llm_decoder(
        args.model,
        device=args.device,
        max_len=int(run_config.get("max_len", 2048)),
        max_creator_candidates=int(run_config.get("max_creator_candidates", 64)),
    )

    (
        qrels_binary,
        qrels_graded,
        run,
        qid_to_query,
        qid_to_labels,
        qid_to_group_ids,
        prediction_rows,
    ) = build_qrels_and_run_from_team_formation_jsonl(
        jsonl_path=jsonl_path,
        decoder=decoder,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.batch_size,
        relevance_gt=args.relevance_gt,
    )

    measures, per_query = evaluate_binary_and_graded(
        qrels_binary=qrels_binary,
        qrels_graded=qrels_graded,
        run=run,
        ks=ks,
        extra_measures=None,
    )
    creator_yaps = load_creator_yaps_from_team_formation_jsonl(jsonl_path)
    group_exposure_per_query = add_group_exposure_metrics_to_per_query(
        per_query=per_query,
        run=run,
        qid_to_labels=qid_to_labels,
        qid_to_group_ids=qid_to_group_ids,
        creator_yaps=creator_yaps,
        k=10,
        kl_target_strategy=str(run_config.get("kl_target_strategy", "high_suppressed_group")),
        alpha_fair=float(run_config.get("alpha_fair", 0.0)),
        delta_high=float(run_config.get("delta_high", 0.05)),
        beta_anti_yap=float(run_config.get("beta_anti_yap", 0.0)),
        tau_relevance=float(run_config.get("tau_relevance", 1.0)),
    )
    if group_exposure_per_query:
        for metric_name in next(iter(group_exposure_per_query.values())).keys():
            measures.add(metric_name)

    within_group_concentration_per_query = compute_within_group_concentration_metrics(
        run=run,
        qid_to_group_ids=qid_to_group_ids,
        ks=ks,
    )
    if within_group_concentration_per_query:
        for qid, metrics in within_group_concentration_per_query.items():
            per_query.setdefault(qid, {}).update(metrics)
        for group_name in ("low", "mid", "high"):
            measures.add(f"within_group_gini_{group_name}_at_10")

    agg = aggregate_metrics(per_query, measures)
    agg.update(
        {
            "run_time": run_config.get("run_time", "unknown"),
            "run_dir": run_config.get("run_dir", str(run_output_dir)),
            "model_dir": str(Path(args.model).resolve()),
            "model_name_or_path": run_config.get("model_name_or_path", args.model),
            "method_name": run_config.get("method_name", "unknown"),
            "loss_type": run_config.get("loss_type", "team_formation_generation"),
            "aux_loss_step_mode": run_config.get("aux_loss_step_mode", "unknown"),
            "adapter_mode": run_config.get("adapter_mode", "unknown"),
            "use_lora": run_config.get("use_lora", False),
            "use_qlora": run_config.get("use_qlora", False),
            "lora_r": run_config.get("lora_r", 0),
            "lora_alpha": run_config.get("lora_alpha", 0),
            "lora_dropout": run_config.get("lora_dropout", 0.0),
            "epochs": run_config.get("epochs", 0),
            "batch_size": run_config.get("batch_size", 0),
            "gradient_accumulation_steps": run_config.get("gradient_accumulation_steps", 0),
            "lr": run_config.get("lr", 0.0),
            "max_len": run_config.get("max_len", decoder.max_len),
            "max_creator_candidates": run_config.get("max_creator_candidates", 64),
            "lambda_ntp": run_config.get("lambda_ntp", 0.0),
            "lambda_kl": run_config.get("lambda_kl", 0.0),
            "lambda_kl_creator": run_config.get("lambda_kl_creator", 0.0),
            "kl_target_strategy": run_config.get("kl_target_strategy", "unknown"),
            "creator_kl_strategy": run_config.get("creator_kl_strategy", "none"),
            "alpha_fair": run_config.get("alpha_fair", 0.0),
            "delta_high": run_config.get("delta_high", 0.05),
            "beta_anti_yap": run_config.get("beta_anti_yap", 0.0),
            "tau_relevance": run_config.get("tau_relevance", 1.0),
        }
    )
    agg = add_yaps_metrics_to_agg(
        agg=agg,
        run=run,
        creator_yaps=creator_yaps,
        ks=ks,
        yaps_lambda=0.0,
        use_yaps=False,
    )
    per_query_csv = per_run_eval_dir / "per_query_metrics.csv"
    results_csv = shared_eval_dir / args.results_csv_name
    predictions_jsonl = per_run_eval_dir / "predictions.jsonl"

    save_metrics_csv(
        per_query=per_query,
        model_name=str(args.model),
        meta=agg,
        output_csv=str(per_query_csv),
        ks=ks,
    )
    append_agg_metrics_csv(
        agg=agg,
        ks=ks,
        output_csv=str(results_csv),
        compare_with_baseline=args.compare_with_baseline,
    )

    with open(predictions_jsonl, "w", encoding="utf-8") as f:
        for row in prediction_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    if args.verbose:
        print("Evaluation complete")
        print(f"Per-query metrics: {per_query_csv}")
        print(f"Results CSV: {results_csv}")
        print(f"Predictions: {predictions_jsonl}")
        for k in ks:
            print(
                f"GradedNDCG@{k}={agg.get(f'graded_ndcg_cut_{k}', 0.0):.4f} "
                f"AvgAttentionScore@{k}={agg.get(f'AvgYaps@{k}', 0.0):.4f} "
                f"DiscAttentionScore@{k}={agg.get(f'DiscYaps@{k}', 0.0):.4f}"
            )
        print(f"Method={agg.get('method_name', 'unknown')}")


if __name__ == "__main__":
    main()
