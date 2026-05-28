from __future__ import annotations


import importlib.util
import json
import os
from typing import Dict, Iterable, List, Sequence

import torch
from torch import nn
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

from loss import (
    normalize_aux_loss_step_mode,
    TeamFormationLoss,
    normalize_kl_target_strategy,
    validate_aux_loss_step_mode,
    validate_loss_type,
)

try:
    from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
except ImportError:
    LoraConfig = None
    PeftModel = None
    get_peft_model = None
    prepare_model_for_kbit_training = None


SUPPORTED_CAUSAL_FAMILIES = ("llama", "qwen", "smollm")
CREATOR_TOKEN_METADATA_FILENAME = "creator_token_metadata.json"
CONFIG_FILENAME = "config.json"
ADAPTER_CONFIG_FILENAME = "adapter_config.json"
MIN_BITSANDBYTES_VERSION = "0.46.1"


def add_creator_tokens(tokenizer, creator_ids: Sequence[str]) -> List[str]:
    creator_tokens = [str(creator_id).strip() for creator_id in creator_ids if str(creator_id).strip()]
    if not creator_tokens:
        raise ValueError("No creator IDs were provided for tokenizer augmentation.")
    tokenizer.add_tokens(creator_tokens, special_tokens=False)
    return creator_tokens


def build_creator_token_maps(tokenizer, creator_ids: Sequence[str]):
    creator_id_to_token = {str(creator_id): str(creator_id) for creator_id in creator_ids}
    creator_token_to_id = {
        token: tokenizer.convert_tokens_to_ids(token)
        for token in creator_id_to_token.values()
    }
    unk_token_id = tokenizer.unk_token_id
    bad_tokens = [
        token
        for token, token_id in creator_token_to_id.items()
        if token_id is None or (unk_token_id is not None and token_id == unk_token_id)
    ]
    if bad_tokens:
        raise ValueError(
            "Some creator IDs were not registered as tokenizer tokens: "
            f"{bad_tokens[:10]}"
        )
    return creator_token_to_id, creator_id_to_token


def get_all_creator_token_ids(creator_token_to_id: Dict[str, int], eos_token_id: int):
    if eos_token_id is None:
        raise ValueError("Tokenizer EOS token must be set for creator generation.")
    ordered = list(creator_token_to_id.values()) + [eos_token_id]
    return torch.tensor(ordered, dtype=torch.long)


def build_allowed_tokens_fn(all_creator_token_ids: Sequence[int]):
    allowed_tokens = [int(token_id) for token_id in all_creator_token_ids]

    def prefix_allowed_tokens_fn(batch_id, input_ids):
        return allowed_tokens

    return prefix_allowed_tokens_fn


def build_project_allowed_tokens_fn(
    project_creator_token_ids: Sequence[Sequence[int]],
    eos_token_id: int,
):
    project_allowed_tokens = [
        [int(token_id) for token_id in token_ids]
        for token_ids in project_creator_token_ids
    ]
    eos_token_id = int(eos_token_id)

    def prefix_allowed_tokens_fn(batch_id, input_ids):
        allowed = project_allowed_tokens[batch_id]
        seen = {int(token_id) for token_id in input_ids if int(token_id) in allowed}
        remaining = [token_id for token_id in allowed if token_id not in seen]
        return remaining + [eos_token_id]

    return prefix_allowed_tokens_fn


def _contains_supported_family(value: str | None) -> bool:
    if not value:
        return False
    lower_value = str(value).lower()
    return any(family in lower_value for family in SUPPORTED_CAUSAL_FAMILIES)


def _iter_local_model_family_hints(model_name_or_path: str) -> List[str]:
    hints: List[str] = []
    model_path = os.path.expanduser(str(model_name_or_path))
    if not os.path.isdir(model_path):
        return hints

    for filename in (CONFIG_FILENAME, ADAPTER_CONFIG_FILENAME):
        config_path = os.path.join(model_path, filename)
        if not os.path.exists(config_path):
            continue
        try:
            with open(config_path, encoding="utf-8") as f:
                config = json.load(f)
        except Exception:
            continue

        for key in ("model_type", "_name_or_path", "base_model_name_or_path"):
            value = config.get(key)
            if isinstance(value, str):
                hints.append(value)

        architectures = config.get("architectures")
        if isinstance(architectures, list):
            hints.extend(str(value) for value in architectures if isinstance(value, str))

    return hints


def _is_supported_causal_model(model_name_or_path: str) -> bool:
    if _contains_supported_family(model_name_or_path):
        return True
    return any(
        _contains_supported_family(hint)
        for hint in _iter_local_model_family_hints(model_name_or_path)
    )


def _load_local_json(path: str) -> Dict | None:
    if not os.path.exists(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _validate_qlora_environment() -> None:
    if BitsAndBytesConfig is None or prepare_model_for_kbit_training is None:
        raise ImportError(
            "QLoRA requires transformers bitsandbytes support and peft to be installed."
        )

    if importlib.util.find_spec("bitsandbytes") is None:
        raise ImportError(
            "QLoRA requires bitsandbytes>=0.46.1, but bitsandbytes is not installed. "
            "Install it with `pip install -U bitsandbytes>=0.46.1`, or switch "
            "`adapter_mode` to `lora` or `none`."
        )


def _load_causal_lm_with_optional_adapter(
    *,
    model_name_or_path: str,
    tokenizer,
    quantization_config,
    attn_implementation,
):
    adapter_config = _load_local_json(os.path.join(model_name_or_path, ADAPTER_CONFIG_FILENAME))
    if adapter_config is not None:
        if PeftModel is None:
            raise ImportError("Loading LoRA/QLoRA adapters requires peft to be installed.")
        base_model_name_or_path = adapter_config.get("base_model_name_or_path")
        if not base_model_name_or_path:
            raise ValueError(
                f"Adapter checkpoint at {model_name_or_path} is missing base_model_name_or_path."
            )
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name_or_path,
            torch_dtype="auto",
            trust_remote_code=True,
            quantization_config=quantization_config,
            attn_implementation=attn_implementation,
        )
        base_model.resize_token_embeddings(len(tokenizer))
        base_model.config.pad_token_id = tokenizer.pad_token_id
        base_model.config.use_cache = False
        return PeftModel.from_pretrained(base_model, model_name_or_path)

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype="auto",
        trust_remote_code=True,
        quantization_config=quantization_config,
        attn_implementation=attn_implementation,
    )
    model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False
    return model


class LLMDecoder(nn.Module):
    def __init__(
        self,
        hf_model=None,
        tokenizer=None,
        creator_ids: Sequence[str] | None = None,
        creator_token_to_id: Dict[str, int] | None = None,
        creator_id_to_token: Dict[str, str] | None = None,
        loss_type="ntp",
        max_len=2048,
        max_creator_candidates=64,
        lambda_ntp=1.0,
        lambda_kl=1.0,
        lambda_kl_creator=0.0,
        aux_loss_step_mode="stepwise",
        kl_target_strategy="none",
        creator_kl_strategy="none",
        alpha_fair=0.2,
        delta_high=0.05,
        beta_anti_yap=1.0,
        tau_relevance=1.0,
        task_type="team_formation",
        use_lora=False,
        use_qlora=False,
    ):
        super().__init__()

        self.model = hf_model
        self.tokenizer = tokenizer
        self.loss_type = validate_loss_type(loss_type)
        self.max_len = max_len
        self.max_creator_candidates = max_creator_candidates
        self.aux_loss_step_mode = normalize_aux_loss_step_mode(
            self.loss_type,
            aux_loss_step_mode,
        )
        self.kl_target_strategy = normalize_kl_target_strategy(self.loss_type, kl_target_strategy)
        self.alpha_fair = float(alpha_fair)
        self.delta_high = float(delta_high)
        self.beta_anti_yap = float(beta_anti_yap)
        self.tau_relevance = float(tau_relevance)
        self.task_type = task_type
        self.use_lora = use_lora
        self.use_qlora = use_qlora

        if creator_token_to_id is None or creator_id_to_token is None:
            if creator_ids is None:
                raise ValueError("creator_ids or saved creator token metadata must be provided.")
            creator_token_to_id, creator_id_to_token = build_creator_token_maps(
                tokenizer=tokenizer,
                creator_ids=creator_ids,
            )

        self.creator_token_to_id = dict(creator_token_to_id)
        self.creator_id_to_token = dict(creator_id_to_token)
        self.creator_ids = sorted(self.creator_id_to_token.keys())
        self.register_buffer(
            "all_creator_token_ids",
            get_all_creator_token_ids(
                creator_token_to_id=self.creator_token_to_id,
                eos_token_id=self.tokenizer.eos_token_id,
            ),
            persistent=False,
        )

        # Generation is restricted to creator IDs plus EOS so the task head never emits
        # arbitrary natural-language tokens after the prompt.
        self.prefix_allowed_tokens_fn = build_allowed_tokens_fn(self.all_creator_token_ids.tolist())

        self.loss_fn = TeamFormationLoss(
            mode=self.loss_type,
            lambda_ntp=lambda_ntp,
            lambda_kl=lambda_kl,
            lambda_kl_creator=lambda_kl_creator,
            aux_loss_step_mode=self.aux_loss_step_mode,
            kl_target_strategy=self.kl_target_strategy,
            creator_kl_strategy=creator_kl_strategy,
            alpha_fair=self.alpha_fair,
            delta_high=self.delta_high,
            beta_anti_yap=self.beta_anti_yap,
            tau_relevance=self.tau_relevance,
            eos_token_id=self.tokenizer.eos_token_id,
        )

    def get_project_creator_token_ids(self, creator_ids: Sequence[str]) -> List[int]:
        token_ids = []
        for creator_id in creator_ids:
            creator_id = str(creator_id)
            if creator_id not in self.creator_id_to_token:
                raise KeyError(f"Unknown creator_id: {creator_id}")
            token = self.creator_id_to_token[creator_id]
            token_id = self.creator_token_to_id[token]
            if token_id == self.tokenizer.eos_token_id:
                raise ValueError("project_creator_token_ids must not include EOS.")
            token_ids.append(token_id)
        return token_ids

    def forward(self, batch, labels=None):
        if self.task_type != "team_formation":
            raise ValueError(f"Unsupported task_type for LLMDecoder: {self.task_type}")

        outputs = self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        )

        losses = self.loss_fn(
            logits=outputs.logits,
            labels=batch["labels"],
            all_creator_token_ids=self.all_creator_token_ids.to(outputs.logits.device),
            project_creator_token_ids=batch["project_creator_token_ids"],
            rank_labels=batch["project_rank_labels"],
            project_yap_scores=batch["project_yap_scores"],
            project_group_ids=batch["project_group_ids"],
        )

        outputs["loss"] = losses["loss"]
        outputs["losses"] = losses
        return outputs

    def preprocess(self, features: List[Dict], max_len=None):
        max_len = max_len or self.max_len
        bos_token_id = self.tokenizer.bos_token_id
        pad_token_id = self.tokenizer.pad_token_id
        eos_token_id = self.tokenizer.eos_token_id

        input_id_rows = []
        label_rows = []
        batch_project_ids = []
        batch_target_creator_ids = []
        batch_target_creator_token_ids = []
        batch_project_creator_ids = []
        batch_project_creator_token_ids = []
        batch_project_rank_labels = []
        batch_project_yap_scores = []
        batch_project_group_ids = []

        for feature in features:
            target_token_ids = list(feature["target_creator_token_ids"])
            if eos_token_id is not None and (not target_token_ids or target_token_ids[-1] != eos_token_id):
                target_token_ids = target_token_ids + [eos_token_id]

            max_prompt_len = max_len - len(target_token_ids) - (1 if bos_token_id is not None else 0)
            max_prompt_len = max(max_prompt_len, 0)
            prompt_ids = self.tokenizer.encode(
                feature["prompt"],
                add_special_tokens=False,
                truncation=True,
                max_length=max_prompt_len,
            ) if max_prompt_len > 0 else []

            input_ids = prompt_ids + target_token_ids
            labels = ([-100] * len(prompt_ids)) + target_token_ids

            if bos_token_id is not None:
                input_ids = [bos_token_id] + input_ids
                labels = [-100] + labels

            input_ids = input_ids[:max_len]
            labels = labels[:max_len]

            first_target_token_id = next(
                (token_id for token_id in labels if token_id != -100),
                None,
            )
            if first_target_token_id is None:
                raise ValueError("Labels must contain at least one creator token.")
            if first_target_token_id != target_token_ids[0]:
                raise ValueError(
                    "Labels must start with the first creator token, not a control token."
                )

            input_id_rows.append(torch.tensor(input_ids, dtype=torch.long))
            label_rows.append(torch.tensor(labels, dtype=torch.long))

            batch_project_ids.append(feature["project_id"])
            batch_target_creator_ids.append(list(feature["target_creator_ids"]))
            batch_target_creator_token_ids.append(list(feature["target_creator_token_ids"]))
            batch_project_creator_ids.append(list(feature["project_creator_ids"]))
            batch_project_creator_token_ids.append(list(feature["project_creator_token_ids"]))
            batch_project_rank_labels.append(list(feature["project_rank_labels"]))
            batch_project_yap_scores.append(list(feature["project_yap_scores"]))
            batch_project_group_ids.append(list(feature["project_group_ids"]))

        batch_size = len(features)
        max_seq_len = max(row.size(0) for row in input_id_rows)
        input_ids = torch.full((batch_size, max_seq_len), pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((batch_size, max_seq_len), dtype=torch.long)
        labels = torch.full((batch_size, max_seq_len), -100, dtype=torch.long)

        for row_idx in range(batch_size):
            seq_len = input_id_rows[row_idx].size(0)
            input_ids[row_idx, :seq_len] = input_id_rows[row_idx]
            attention_mask[row_idx, :seq_len] = 1
            labels[row_idx, :seq_len] = label_rows[row_idx]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "project_id": batch_project_ids,
            "target_creator_ids": batch_target_creator_ids,
            "target_creator_token_ids": batch_target_creator_token_ids,
            "project_creator_ids": batch_project_creator_ids,
            "project_creator_token_ids": batch_project_creator_token_ids,
            "project_rank_labels": batch_project_rank_labels,
            "project_yap_scores": batch_project_yap_scores,
            "project_group_ids": batch_project_group_ids,
        }

    def generate_creator_ids(self, prompts, max_new_tokens=32, **generate_kwargs):
        if isinstance(prompts, str):
            prompts = [prompts]
        project_creator_ids = generate_kwargs.pop("project_creator_ids", None)
        model_inputs = self.tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt",
        )
        model_inputs = {k: v.to(self.model.device) for k, v in model_inputs.items()}

        prefix_allowed_tokens_fn = self.prefix_allowed_tokens_fn
        if project_creator_ids is not None:
            if len(project_creator_ids) != len(prompts):
                raise ValueError(
                    "project_creator_ids must align with prompts for constrained generation."
                )
            project_creator_token_ids = [
                self.get_project_creator_token_ids(sample_creator_ids)
                for sample_creator_ids in project_creator_ids
            ]
            prefix_allowed_tokens_fn = build_project_allowed_tokens_fn(
                project_creator_token_ids=project_creator_token_ids,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        outputs = self.model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            **generate_kwargs,
        )

        prompt_len = model_inputs["input_ids"].size(1)
        generated_ids = outputs[:, prompt_len:]
        decoded = []
        for row in generated_ids.tolist():
            creator_ids = []
            for token_id in row:
                if token_id == self.tokenizer.eos_token_id:
                    break
                creator_ids.append(self.tokenizer.decode([token_id], skip_special_tokens=False).strip())
            decoded.append(creator_ids)
        return decoded

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path,
        creator_ids: Sequence[str] | None = None,
        loss_type="ntp",
        max_len=2048,
        max_creator_candidates=64,
        lambda_ntp=1.0,
        lambda_kl=1.0,
        lambda_kl_creator=0.0,
        aux_loss_step_mode="stepwise",
        kl_target_strategy="none",
        creator_kl_strategy="none",
        alpha_fair=0.2,
        delta_high=0.05,
        beta_anti_yap=1.0,
        tau_relevance=1.0,
        task_type="team_formation",
        use_lora=False,
        use_qlora=False,
        lora_r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        lora_target_modules=None,
        attn_implementation=None,
    ):
        if not _is_supported_causal_model(model_name_or_path):
            raise ValueError(
                "Current team-formation implementation supports Llama and Qwen causal "
                "LLMs. Please use a Llama 3 Instruct or Qwen 2.5 Instruct checkpoint."
            )

        tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            use_fast=True,
            trust_remote_code=True,
        )
        if hasattr(tokenizer, "deprecation_warnings"):
            tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = True
        tokenizer.padding_side = "right"
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        creator_token_to_id = None
        creator_id_to_token = None
        loaded_from_saved_metadata = False
        metadata_path = os.path.join(model_name_or_path, CREATOR_TOKEN_METADATA_FILENAME)
        if os.path.exists(metadata_path):
            with open(metadata_path, encoding="utf-8") as f:
                metadata = json.load(f)
            creator_token_to_id = metadata["creator_token_to_id"]
            creator_id_to_token = metadata["creator_id_to_token"]
            creator_ids = list(creator_id_to_token.keys())
            loaded_from_saved_metadata = True

        if creator_ids is None:
            raise ValueError(
                "creator_ids must be provided when loading a base model without saved creator metadata."
            )

        creator_ids = sorted({str(creator_id).strip() for creator_id in creator_ids if str(creator_id).strip()})
        if not loaded_from_saved_metadata:
            add_creator_tokens(tokenizer, creator_ids)

        quantization_config = None
        if use_qlora:
            _validate_qlora_environment()
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )

        model = _load_causal_lm_with_optional_adapter(
            model_name_or_path=model_name_or_path,
            tokenizer=tokenizer,
            quantization_config=quantization_config,
            attn_implementation=attn_implementation,
        )

        if use_qlora:
            model = prepare_model_for_kbit_training(model)

        if use_lora or use_qlora:
            if LoraConfig is None or get_peft_model is None:
                raise ImportError("LoRA/QLoRA requires peft to be installed.")
            if lora_target_modules is None:
                lora_target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
            peft_config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=lora_target_modules,
                bias="none",
                task_type="CAUSAL_LM",
            )
            model = get_peft_model(model, peft_config)

        reranker = cls(
            hf_model=model,
            tokenizer=tokenizer,
            creator_ids=creator_ids,
            creator_token_to_id=creator_token_to_id,
            creator_id_to_token=creator_id_to_token,
            loss_type=loss_type,
            max_len=max_len,
            max_creator_candidates=max_creator_candidates,
            lambda_ntp=lambda_ntp,
            lambda_kl=lambda_kl,
            lambda_kl_creator=lambda_kl_creator,
            aux_loss_step_mode=aux_loss_step_mode,
            kl_target_strategy=kl_target_strategy,
            creator_kl_strategy=creator_kl_strategy,
            alpha_fair=alpha_fair,
            delta_high=delta_high,
            beta_anti_yap=beta_anti_yap,
            tau_relevance=tau_relevance,
            task_type=task_type,
            use_lora=use_lora,
            use_qlora=use_qlora,
        )
        return reranker

    def save_pretrained(self, save_dir, safe_serialization=False):
        os.makedirs(save_dir, exist_ok=True)
        self.model.save_pretrained(save_dir, safe_serialization=safe_serialization)
        metadata = {
            "creator_token_to_id": self.creator_token_to_id,
            "creator_id_to_token": self.creator_id_to_token,
            "all_creator_token_ids": self.all_creator_token_ids.tolist(),
        }
        with open(
            os.path.join(save_dir, CREATOR_TOKEN_METADATA_FILENAME),
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(metadata, f, ensure_ascii=True, indent=2, sort_keys=True)
