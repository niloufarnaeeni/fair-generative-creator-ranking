import os
from datetime import datetime
from pathlib import Path
import json
import shutil
from zoneinfo import ZoneInfo
from typing import Any

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
from accelerate.utils import set_seed, ProjectConfiguration
from transformers import get_wsd_schedule
from loss import (
    loss_uses_aux_loss_step_mode,
    loss_uses_kl_target_strategy,
    normalize_aux_loss_step_mode,
    validate_aux_loss_step_mode,
    validate_creator_kl_strategy,
    validate_kl_target_strategy,
    validate_loss_type,
)
from data import (
    TeamFormationDataset,
    collect_creator_ids_from_creators_csv,
    collect_creator_ids_from_jsonl,
)
from torch.utils.data import DataLoader
from trainer import Trainer
from accelerate import Accelerator


PAPER_METHODS = ("ntp", "fairgroup", "suppgroup", "suppgroup_creator")


def infer_paper_method_name(config: dict[str, Any]) -> str:
    explicit_method = str(config.get("method_name", "") or "").strip().lower()
    if explicit_method in PAPER_METHODS:
        return explicit_method
    loss_type = str(config.get("loss_type", "ntp"))
    kl_target_strategy = str(config.get("kl_target_strategy", "none"))
    creator_kl_strategy = str(config.get("creator_kl_strategy", "none"))
    lambda_kl_creator = float(config.get("lambda_kl_creator", 0.0) or 0.0)

    if loss_type == "ntp":
        return "ntp"
    if (
        loss_type == "ntp_kl"
        and kl_target_strategy == "fair_group"
        and creator_kl_strategy == "none"
    ):
        return "fairgroup"
    if (
        loss_type == "ntp_kl"
        and kl_target_strategy == "high_suppressed_group"
        and creator_kl_strategy == "none"
    ):
        return "suppgroup"
    if (
        loss_type == "ntp_kl"
        and kl_target_strategy == "high_suppressed_group"
        and creator_kl_strategy == "creator_anti_attention"
        and lambda_kl_creator != 0.0
    ):
        return "suppgroup_creator"
    raise ValueError(
        "Could not infer a paper method from the current config. "
        "Use one of method_name={ntp,fairgroup,suppgroup,suppgroup_creator}."
    )


def apply_paper_method_config(args) -> None:
    method_name = str(getattr(args, "method_name", "") or "").strip().lower()
    if not method_name:
        method_name = infer_paper_method_name(vars(args))
    if method_name not in PAPER_METHODS:
        supported = ", ".join(PAPER_METHODS)
        raise ValueError(f"Unsupported method_name '{method_name}'. Supported values: {supported}")

    args.method_name = method_name
    args.aux_loss_step_mode = "none" if method_name == "ntp" else "stepwise"

    if method_name == "ntp":
        args.loss_type = "ntp"
        args.kl_target_strategy = "none"
        args.creator_kl_strategy = "none"
        args.lambda_kl = 0.0
        args.lambda_kl_creator = 0.0
        args.alpha_fair = 0.0
        args.delta_high = 0.0
        args.beta_anti_yap = 0.0
        args.tau_relevance = 1.0
        return

    args.loss_type = "ntp_kl"
    args.lambda_kl = float(args.lambda_kl)

    if method_name == "fairgroup":
        args.kl_target_strategy = "fair_group"
        args.creator_kl_strategy = "none"
        args.lambda_kl_creator = 0.0
        args.delta_high = 0.0
        args.beta_anti_yap = 0.0
        args.tau_relevance = 1.0
        args.alpha_fair = float(args.alpha_fair)
        return

    args.kl_target_strategy = "high_suppressed_group"
    args.alpha_fair = 0.0
    args.delta_high = float(args.delta_high)

    if method_name == "suppgroup":
        args.creator_kl_strategy = "none"
        args.lambda_kl_creator = 0.0
        args.beta_anti_yap = 0.0
        args.tau_relevance = 1.0
        return

    args.creator_kl_strategy = "creator_anti_attention"
    args.lambda_kl_creator = float(args.lambda_kl_creator)
    if args.lambda_kl_creator == 0.0:
        raise ValueError("suppgroup_creator requires lambda_kl_creator > 0.")
    # beta_anti_yap controls the strength of the creator_anti_attention downweighting.
    args.beta_anti_yap = float(args.beta_anti_yap)
    args.tau_relevance = float(args.tau_relevance)


def _infer_dataset_name(train_dataset: str | None) -> str:
    if not train_dataset:
        return "dataset"
    path = Path(train_dataset).resolve()
    parts = list(path.parts)

    rag_data_indices = [
        idx for idx in range(len(parts) - 1)
        if parts[idx] == "data" and idx > 0 and parts[idx - 1] == "rag_retrieval"
    ]
    if rag_data_indices:
        idx = rag_data_indices[-1]
        return parts[idx + 1]

    data_indices = [idx for idx, part in enumerate(parts[:-1]) if part == "data"]
    if data_indices:
        idx = data_indices[-1]
        return parts[idx + 1]

    if len(parts) >= 2:
        return parts[-2]
    return path.stem or "dataset"


def _build_run_output_dir(base_output_dir: str, train_dataset: str | None, run_name: str | None = None) -> str:
    base_path = Path(base_output_dir).resolve()
    root_dir = base_path if base_path.name == "output" else base_path.parent
    dataset_name = _infer_dataset_name(train_dataset)
    run_dir_name = datetime.now(ZoneInfo("America/Toronto")).strftime("%Y_%m_%d_%H_%M_%S")
    return str(root_dir / dataset_name / "run" / run_dir_name)


def _resolve_config_path(config_path: str | None, value: str | None) -> str | None:
    if value is None or not isinstance(value, str):
        return value
    if value.strip() == "":
        return value
    path = Path(value)
    if path.is_absolute():
        return str(path)
    if config_path is None:
        return value
    resolved = (Path(config_path).resolve().parent / path).resolve()
    return str(resolved)


def _resolve_output_dir_path(value: str | None) -> str | None:
    if value is None or not isinstance(value, str):
        return value
    if value.strip() == "":
        return value
    path = Path(value)
    if path.is_absolute():
        return str(path)
    return str((Path.cwd().resolve() / path).resolve())


def _infer_raw_creators_csv_path(train_dataset: str | None) -> str | None:
    if not train_dataset:
        return None
    path = Path(train_dataset).resolve()
    parts = list(path.parts)
    if "data" not in parts:
        return None
    data_idx = parts.index("data")
    if data_idx + 1 >= len(parts):
        return None
    dataset_root = Path(*parts[: data_idx + 2])
    raw_creators_csv = dataset_root / "raw" / "creators.csv"
    return str(raw_creators_csv)


def _sanitize_tracker_config(config: dict) -> dict:
    sanitized = {}
    for key, value in config.items():
        if isinstance(value, (int, float, str, bool)) or value is None:
            sanitized[key] = value if value is not None else "None"
        else:
            sanitized[key] = str(value)
    return sanitized


def _write_run_metadata(args) -> None:
    run_dir = Path(args.output_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    run_config = _sanitize_tracker_config(vars(args))
    run_config["run_dir"] = str(run_dir)
    run_config["run_time"] = run_dir.name

    with open(run_dir / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(run_config, f, ensure_ascii=True, indent=2, sort_keys=True)

    try:
        import yaml

        with open(run_dir / "resolved_config.yaml", "w", encoding="utf-8") as f:
            yaml.safe_dump(vars(args), f, sort_keys=False)
    except Exception:
        pass

    if getattr(args, "config", None):
        config_path = Path(args.config)
        if config_path.exists():
            shutil.copy2(config_path, run_dir / "source_config.yaml")


def _normalize_adapter_settings(args) -> None:
    adapter_mode = getattr(args, "adapter_mode", None)

    if adapter_mode == "lora":
        args.use_lora = 1
        args.use_qlora = 0
    elif adapter_mode == "qlora":
        args.use_lora = 0
        args.use_qlora = 1
    else:
        args.use_lora = 0
        args.use_qlora = 0

    if not getattr(args, "adapter_mode", None):
        if bool(getattr(args, "use_qlora", 0)):
            args.adapter_mode = "qlora"
        elif bool(getattr(args, "use_lora", 0)):
            args.adapter_mode = "lora"
        else:
            args.adapter_mode = "none"


def create_adamw_optimizer(
    model,
    lr,
    weight_decay = 1e-2,
    no_decay_keywords = ('bias', 'LayerNorm', 'layernorm'),
):
    parameters = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    optimizer_grouped_parameters = [
        {
            'params': [p for n, p in parameters if not any(nd in n for nd in no_decay_keywords)],
            'weight_decay': weight_decay,
        },
        {
            'params': [p for n, p in parameters if any(nd in n for nd in no_decay_keywords)],
            'weight_decay': 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=lr)
    return optimizer


def parse_args():
    import yaml
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str)
    parser.add_argument("--model_name_or_path", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--train_dataset", help="training file")
    parser.add_argument("--val_dataset", help="validation file", default=None)
    parser.add_argument("--output_dir", help="output dir", default="./output")
    parser.add_argument("--save_on_epoch_end", type=int, default=0)
    parser.add_argument("--num_max_checkpoints", type=int, default=5)
    parser.add_argument("--max_len", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--early_stopping_patience", type=int, default=2)
    parser.add_argument("--early_stopping_min_delta", type=float, default=0.0)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=666)
    parser.add_argument("--warmup_proportion", type=float, default=0.1)
    parser.add_argument("--stable_proportion", type=float, default=0.0)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument(
        "--method_name",
        type=str,
        default=None,
        help="Paper method name: ntp, fairgroup, suppgroup, or suppgroup_creator.",
    )
    parser.add_argument(
        "--loss_type",
        type=str,
        default="ntp",
        help="Internal loss type. Paper-facing runs should use method_name instead.",
    )
    parser.add_argument(
        "--log_with", type=str, default="wandb", help="wandb, tensorboard"
    )
    parser.add_argument("--mixed_precision", type=str, default=None)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--task_type", type=str, default="team_formation")
    parser.add_argument("--max_creator_candidates", type=int, default=64)
    parser.add_argument("--lambda_ntp", type=float, default=1.0)
    parser.add_argument("--lambda_kl", type=float, default=1.0)
    parser.add_argument("--lambda_kl_creator", type=float, default=0.0)
    parser.add_argument("--aux_loss_step_mode", type=str, default="stepwise")
    parser.add_argument("--kl_target_strategy", type=str, default="none")
    parser.add_argument("--creator_kl_strategy", type=str, default="none")
    parser.add_argument("--alpha_fair", type=float, default=0.2)
    parser.add_argument("--delta_high", type=float, default=0.05)
    parser.add_argument("--beta_anti_yap", type=float, default=1.0)
    parser.add_argument("--tau_relevance", type=float, default=1.0)
    parser.add_argument("--creator_id_key", type=str, default="creator_id")
    parser.add_argument("--creator_yap_key", type=str, default="yap_score")
    parser.add_argument("--project_name_key", type=str, default="project_name")
    parser.add_argument("--project_description_key", type=str, default="project_description")
    parser.add_argument("--required_skills_key", type=str, default="required_skills")
    parser.add_argument("--creators_key", type=str, default="creators")
    parser.add_argument("--raw_creators_csv", type=str, default=None)
    parser.add_argument("--team_key", type=str, default="team_creator_ids")
    parser.add_argument("--creator_rank_key", type=str, default="rank")
    parser.add_argument("--use_lora", type=int, default=0)
    parser.add_argument("--use_qlora", type=int, default=0)
    parser.add_argument("--adapter_mode", type=str, default="none")
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--lora_target_modules", nargs="*", default=None)
    parser.add_argument("--attn_implementation", type=str, default=None)
    parser.add_argument("--run_name", type=str, default=None)

    args = parser.parse_args()

    # 加载 YAML 配置文件
    with open(args.config, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    # 使用 YAML 配置文件中的参数覆盖命令行参数
    for key, value in config.items():
        setattr(args, key, value)

    path_like_keys = [
        "train_dataset",
        "val_dataset",
        "raw_creators_csv",
    ]
    for key in path_like_keys:
        if hasattr(args, key):
            setattr(args, key, _resolve_config_path(args.config, getattr(args, key)))

    if hasattr(args, "output_dir"):
        args.output_dir = _resolve_output_dir_path(args.output_dir)

    _normalize_adapter_settings(args)
    apply_paper_method_config(args)

    return args


def main():

    args = parse_args()
    if args.task_type != "team_formation":
        raise ValueError("Only task_type='team_formation' is supported in this release.")

    validate_loss_type(args.loss_type)
    if loss_uses_aux_loss_step_mode(args.loss_type):
        validate_aux_loss_step_mode(args.aux_loss_step_mode)
    else:
        args.aux_loss_step_mode = normalize_aux_loss_step_mode(
            args.loss_type,
            args.aux_loss_step_mode,
        )
    if loss_uses_kl_target_strategy(args.loss_type):
        validate_kl_target_strategy(args.kl_target_strategy)
        if float(args.delta_high) < 0.0:
            raise ValueError("delta_high must be non-negative.")
        if float(args.beta_anti_yap) < 0.0:
            raise ValueError("beta_anti_yap must be non-negative.")
        if float(args.tau_relevance) < 0.0:
            raise ValueError("tau_relevance must be non-negative.")
    else:
        args.kl_target_strategy = "none"
    validate_creator_kl_strategy(args.creator_kl_strategy)
    args.lambda_kl_creator = float(args.lambda_kl_creator)
    if args.lambda_kl_creator == 0.0:
        args.creator_kl_strategy = "none"
    if (
        args.creator_kl_strategy == "creator_anti_attention"
        and float(args.beta_anti_yap) < 0.0
    ):
        raise ValueError("beta_anti_yap must be non-negative.")
    if (
        args.creator_kl_strategy == "creator_anti_attention"
        and float(args.tau_relevance) < 0.0
    ):
        raise ValueError("tau_relevance must be non-negative.")

    if args.kl_target_strategy == "fair_group":
        args.alpha_fair = float(args.alpha_fair)
    else:
        args.alpha_fair = 0.0

    if args.kl_target_strategy == "high_suppressed_group":
        args.delta_high = float(args.delta_high)
    else:
        args.delta_high = 0.0

    if args.creator_kl_strategy == "creator_anti_attention":
        args.beta_anti_yap = float(args.beta_anti_yap)
        args.tau_relevance = float(args.tau_relevance)
    else:
        args.beta_anti_yap = 0.0
        args.tau_relevance = 1.0

    set_seed(args.seed)
    args.output_dir = _build_run_output_dir(args.output_dir, args.train_dataset, getattr(args, "run_name", None))
    _write_run_metadata(args)

    project_config = ProjectConfiguration(
        project_dir=str(args.output_dir),
        automatic_checkpoint_naming=True,
        total_limit=args.num_max_checkpoints,
        logging_dir=str(Path(args.output_dir) / "logs"),
    )

    accelerator = Accelerator(
        project_config=project_config,
        log_with=args.log_with,
        mixed_precision=args.mixed_precision,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )

    accelerator.init_trackers("ranker", config=_sanitize_tracker_config(vars(args)))
    accelerator.print(f"Train Args from User Input: {vars(args)}")
    accelerator.print(f"Training outputs will be saved under: {args.output_dir}")
    accelerator.print(f"RUN_DIR::{args.output_dir}")

    raw_creators_csv = args.raw_creators_csv or _infer_raw_creators_csv_path(args.train_dataset)
    creator_ids = []
    if raw_creators_csv and Path(raw_creators_csv).exists():
        creator_ids = collect_creator_ids_from_creators_csv(raw_creators_csv)
        accelerator.print(
            f"Loaded {len(creator_ids)} unique creator IDs from raw creators.csv for tokenizer augmentation."
        )
    if not creator_ids:
        creator_ids = collect_creator_ids_from_jsonl(
            [args.train_dataset, args.val_dataset],
            creators_key=args.creators_key,
            creator_id_key=args.creator_id_key,
            team_key=args.team_key,
        )
        accelerator.print(
            f"Loaded {len(creator_ids)} unique creator IDs from train/valid JSONL for tokenizer augmentation."
        )

    from model_llm import LLMDecoder

    model = LLMDecoder.from_pretrained(
        model_name_or_path=args.model_name_or_path,
        creator_ids=creator_ids,
        loss_type=args.loss_type,
        max_len=args.max_len,
        max_creator_candidates=args.max_creator_candidates,
        lambda_ntp=args.lambda_ntp,
        lambda_kl=args.lambda_kl,
        lambda_kl_creator=args.lambda_kl_creator,
        aux_loss_step_mode=args.aux_loss_step_mode,
        kl_target_strategy=args.kl_target_strategy,
        creator_kl_strategy=args.creator_kl_strategy,
        alpha_fair=args.alpha_fair,
        delta_high=args.delta_high,
        beta_anti_yap=args.beta_anti_yap,
        tau_relevance=args.tau_relevance,
        task_type=args.task_type,
        use_lora=bool(args.use_lora),
        use_qlora=bool(args.use_qlora),
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_target_modules=args.lora_target_modules,
        attn_implementation=args.attn_implementation,
    )

    train_dataset = TeamFormationDataset(
        data_path=args.train_dataset,
        target_model=model,
        max_len=args.max_len,
        tag="training",
        task_type=args.task_type,
        creator_id_key=args.creator_id_key,
        creator_yap_key=args.creator_yap_key,
        project_name_key=args.project_name_key,
        project_description_key=args.project_description_key,
        required_skills_key=args.required_skills_key,
        creators_key=args.creators_key,
        team_key=args.team_key,
        creator_label_key="label",
        creator_rank_key=args.creator_rank_key,
    )

    val_dataset = None
    if args.val_dataset:
        val_dataset = TeamFormationDataset(
            data_path=args.val_dataset,
            target_model=model,
            max_len=args.max_len,
            tag="validation",
            task_type=args.task_type,
            creator_id_key=args.creator_id_key,
            creator_yap_key=args.creator_yap_key,
            project_name_key=args.project_name_key,
            project_description_key=args.project_description_key,
            required_skills_key=args.required_skills_key,
            creators_key=args.creators_key,
            team_key=args.team_key,
            creator_label_key="label",
            creator_rank_key=args.creator_rank_key,
        )

    available_cpus = os.cpu_count() or 1
    slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK")
    if slurm_cpus is not None:
        try:
            available_cpus = min(available_cpus, max(1, int(slurm_cpus)))
        except ValueError:
            pass
    num_workers = min(10, available_cpus)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        collate_fn=train_dataset.collate_fn,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,
    )
    
    val_dataloader = None
    if args.val_dataset:
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            collate_fn=val_dataset.collate_fn,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

    optimizer = create_adamw_optimizer(
        model, lr=float(args.lr)
    )
    assert 0 <= args.warmup_proportion <= 1
    assert 0 <= args.stable_proportion <= 1
    assert args.warmup_proportion + args.stable_proportion <= 1
    total_steps = (
        len(train_dataloader) * args.epochs
    ) // accelerator.gradient_state.num_steps
    num_warmup_steps = int(args.warmup_proportion * total_steps)
    num_stable_steps = int(args.stable_proportion * total_steps)
    
    # lr_scheduler = get_cosine_schedule_with_warmup(
    #     optimizer=optimizer,
    #     num_warmup_steps=num_warmup_steps,
    #     num_training_steps=total_steps,
    # )
    lr_scheduler = get_wsd_schedule(
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_stable_steps=num_stable_steps,
        num_decay_steps=total_steps - num_warmup_steps - num_stable_steps
    )


    model, optimizer, lr_scheduler, train_dataloader, val_dataloader = (
        accelerator.prepare(
            model, optimizer, lr_scheduler, train_dataloader, val_dataloader
        )
    )

    accelerator.wait_for_everyone()

    trainer = Trainer(
        model=model,
        tokenizer=model.tokenizer,
        optimizer=optimizer,
        train_dataloader=train_dataloader,
        validation_dataloader=val_dataloader,
        accelerator=accelerator,
        epochs=args.epochs,
        lr_scheduler=lr_scheduler,
        log_interval=args.log_interval * accelerator.gradient_state.num_steps,
        save_on_epoch_end=args.save_on_epoch_end,
        early_stopping_patience=(
            args.early_stopping_patience if args.val_dataset else None
        ),
        early_stopping_min_delta=args.early_stopping_min_delta,
    )

    accelerator.print(f"Start training for {args.epochs} epochs ...")
    trainer.train()
    accelerator.print("Training finished!")

    accelerator.print("Saving model ...")
    save_dir = args.output_dir + "/model"
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.save_pretrained(save_dir, safe_serialization=False)
    model.tokenizer.save_pretrained(save_dir)
    accelerator.print("Saving Successfully!")


if __name__ == "__main__":
    main()
