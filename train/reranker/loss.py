from __future__ import annotations

from typing import List

import torch
import torch.nn.functional as F


IGNORE_INDEX = -100
VALID_LOSS_TYPES = {"ntp", "ntp_kl"}
VALID_AUX_LOSS_STEP_MODES = {"stepwise"}
VALID_CREATOR_KL_STRATEGIES = {"none", "creator_anti_attention"}
VALID_KL_TARGET_STRATEGIES = {"fair_group", "high_suppressed_group"}
NUM_YAP_GROUPS = 3


def validate_loss_type(loss_type: str) -> str:
    if loss_type not in VALID_LOSS_TYPES:
        supported = ", ".join(sorted(VALID_LOSS_TYPES))
        raise ValueError(f"Unsupported loss_type '{loss_type}'. Supported values: {supported}")
    return loss_type


def validate_aux_loss_step_mode(aux_loss_step_mode: str) -> str:
    if aux_loss_step_mode not in VALID_AUX_LOSS_STEP_MODES:
        supported = ", ".join(sorted(VALID_AUX_LOSS_STEP_MODES))
        raise ValueError(
            "Unsupported aux_loss_step_mode "
            f"'{aux_loss_step_mode}'. Supported values: {supported}"
        )
    return aux_loss_step_mode


def validate_kl_target_strategy(kl_target_strategy: str) -> str:
    if kl_target_strategy not in VALID_KL_TARGET_STRATEGIES:
        supported = ", ".join(sorted(VALID_KL_TARGET_STRATEGIES))
        raise ValueError(
            "Unsupported kl_target_strategy "
            f"'{kl_target_strategy}'. Supported values: {supported}"
        )
    return kl_target_strategy


def validate_creator_kl_strategy(creator_kl_strategy: str) -> str:
    if creator_kl_strategy not in VALID_CREATOR_KL_STRATEGIES:
        supported = ", ".join(sorted(VALID_CREATOR_KL_STRATEGIES))
        raise ValueError(
            "Unsupported creator_kl_strategy "
            f"'{creator_kl_strategy}'. Supported values: {supported}"
        )
    return creator_kl_strategy


def loss_uses_aux_loss_step_mode(loss_type: str) -> bool:
    return "kl" in loss_type


def normalize_aux_loss_step_mode(loss_type: str, aux_loss_step_mode: str | None) -> str:
    if not loss_uses_aux_loss_step_mode(loss_type):
        return "none"
    return validate_aux_loss_step_mode(str(aux_loss_step_mode or "stepwise"))


def loss_uses_kl_target_strategy(loss_type: str) -> bool:
    return "kl" in loss_type


def normalize_kl_target_strategy(loss_type: str, kl_target_strategy: str | None) -> str:
    if not loss_uses_kl_target_strategy(loss_type):
        return "none"
    return validate_kl_target_strategy(str(kl_target_strategy or "high_suppressed_group"))


def normalize_creator_kl_strategy(creator_kl_strategy: str | None) -> str:
    return validate_creator_kl_strategy(str(creator_kl_strategy or "none"))


def _get_shifted_logits_and_labels(logits, labels):
    return logits[..., :-1, :].contiguous(), labels[..., 1:].contiguous()


def get_creator_step_positions(labels, ignore_index=IGNORE_INDEX):
    shift_labels = labels[..., 1:].contiguous()
    valid_positions = shift_labels.ne(ignore_index)
    has_valid_target = valid_positions.any(dim=1)
    creator_step_positions = [
        torch.nonzero(sample_valid_positions, as_tuple=False).squeeze(-1)
        for sample_valid_positions in valid_positions
    ]
    return shift_labels, creator_step_positions, has_valid_target


def compute_restricted_ntp_loss(
    logits,
    labels,
    allowed_token_ids,
    ignore_index=IGNORE_INDEX,
):
    """Compute creator-generation NTP over all dataset creator tokens plus EOS only."""
    shift_logits, shift_labels = _get_shifted_logits_and_labels(logits, labels)
    valid_positions = shift_labels.ne(ignore_index)
    if not valid_positions.any():
        return logits.new_zeros(())

    allowed_token_ids = allowed_token_ids.to(logits.device)
    restricted_logits = shift_logits.index_select(dim=-1, index=allowed_token_ids)

    vocab_size = logits.size(-1)
    full_to_restricted = torch.full(
        (vocab_size,),
        fill_value=-1,
        dtype=torch.long,
        device=logits.device,
    )
    full_to_restricted[allowed_token_ids] = torch.arange(
        allowed_token_ids.numel(),
        device=logits.device,
    )

    safe_labels = shift_labels.masked_fill(~valid_positions, 0)
    restricted_labels = full_to_restricted[safe_labels]
    invalid_targets = valid_positions & restricted_labels.lt(0)
    if invalid_targets.any():
        bad_ids = torch.unique(shift_labels[invalid_targets]).tolist()
        raise ValueError(
            "Restricted NTP received labels outside creator-token space. "
            f"Unexpected token ids: {bad_ids}"
        )

    flat_logits = restricted_logits[valid_positions]
    flat_labels = restricted_labels[valid_positions]
    return F.cross_entropy(flat_logits, flat_labels)


def _build_group_memberships(creator_scores, group_ids):
    groups = torch.as_tensor(group_ids, device=creator_scores.device)
    if groups.ndim == 1:
        if groups.numel() != creator_scores.numel():
            raise ValueError("group ids must align with project creators.")
        groups = groups.to(dtype=torch.long)
        return F.one_hot(groups, num_classes=NUM_YAP_GROUPS).to(dtype=creator_scores.dtype)
    if groups.ndim == 2:
        if groups.shape != (creator_scores.numel(), NUM_YAP_GROUPS):
            raise ValueError(
                "Soft yap-group memberships must have shape "
                f"[num_creators, {NUM_YAP_GROUPS}]."
            )
        memberships = groups.to(dtype=creator_scores.dtype)
        row_sums = memberships.sum(dim=-1, keepdim=True)
        if torch.any(row_sums <= 0):
            raise ValueError("Soft yap-group memberships must have positive row sums.")
        return memberships / row_sums
    raise ValueError("group_ids must be either hard group ids or per-group memberships.")


def _compute_relevance_group_mass(creator_scores, relevance_labels, group_ids):
    relevance = torch.as_tensor(
        relevance_labels,
        dtype=creator_scores.dtype,
        device=creator_scores.device,
    )
    if creator_scores.numel() != relevance.numel():
        raise ValueError("relevance labels and group ids must align with project creators.")

    memberships = _build_group_memberships(creator_scores, group_ids)
    total_relevance_mass = relevance.sum()
    if total_relevance_mass <= 0:
        return None, memberships

    epsilon_rel = (relevance.unsqueeze(-1) * memberships).sum(dim=0) / total_relevance_mass
    return epsilon_rel, memberships


def _compute_fair_group_target(epsilon_rel, alpha_fair):
    uniform = torch.full(
        (NUM_YAP_GROUPS,),
        fill_value=1.0 / NUM_YAP_GROUPS,
        dtype=epsilon_rel.dtype,
        device=epsilon_rel.device,
    )
    return (1.0 - float(alpha_fair)) * epsilon_rel + float(alpha_fair) * uniform


def _compute_high_suppressed_group_target(epsilon_rel, delta_high):
    target = epsilon_rel.clone()
    delta = torch.as_tensor(
        float(delta_high),
        dtype=epsilon_rel.dtype,
        device=epsilon_rel.device,
    )
    target[2] = torch.clamp(target[2] - delta, min=0.0)
    half_delta = delta / 2.0
    target[0] = target[0] + half_delta
    target[1] = target[1] + half_delta
    target_sum = target.sum()
    if target_sum <= 0:
        return None
    return target / target_sum


def _compute_group_target_distribution(
    creator_scores,
    relevance_labels,
    group_ids,
    kl_target_strategy,
    alpha_fair,
    delta_high,
):
    strategy = validate_kl_target_strategy(kl_target_strategy)
    epsilon_rel, memberships = _compute_relevance_group_mass(
        creator_scores=creator_scores,
        relevance_labels=relevance_labels,
        group_ids=group_ids,
    )
    if epsilon_rel is None:
        return None, memberships

    if strategy == "fair_group":
        return _compute_fair_group_target(epsilon_rel, alpha_fair), memberships
    return _compute_high_suppressed_group_target(epsilon_rel, delta_high), memberships


def _compute_group_kl_from_scores(
    creator_scores,
    relevance_labels,
    group_ids,
    kl_target_strategy,
    alpha_fair,
    delta_high,
    eps=1e-8,
):
    epsilon_target, memberships = _compute_group_target_distribution(
        creator_scores=creator_scores,
        relevance_labels=relevance_labels,
        group_ids=group_ids,
        kl_target_strategy=kl_target_strategy,
        alpha_fair=alpha_fair,
        delta_high=delta_high,
    )
    if epsilon_target is None:
        return None

    creator_probs = F.softmax(creator_scores, dim=-1)
    epsilon_pred = (creator_probs.unsqueeze(-1) * memberships).sum(dim=0)
    valid_groups = epsilon_target.gt(0)
    if not valid_groups.any():
        return None

    safe_target = epsilon_target[valid_groups].clamp_min(eps)
    safe_pred = epsilon_pred[valid_groups].clamp_min(eps)
    return torch.sum(safe_target * (torch.log(safe_target) - torch.log(safe_pred)))


def _compute_creator_anti_attention_kl_from_scores(
    creator_scores,
    relevance_labels,
    yap_scores,
    group_ids,
    kl_target_strategy,
    alpha_fair,
    delta_high,
    beta_anti_yap,
    tau_relevance,
    eps=1e-8,
):
    relevance = torch.as_tensor(
        relevance_labels,
        dtype=creator_scores.dtype,
        device=creator_scores.device,
    )
    yap = torch.as_tensor(
        yap_scores,
        dtype=creator_scores.dtype,
        device=creator_scores.device,
    )
    if creator_scores.numel() != relevance.numel() or creator_scores.numel() != yap.numel():
        raise ValueError("relevance labels, yap scores, and group ids must align with project creators.")

    epsilon_target, memberships = _compute_group_target_distribution(
        creator_scores=creator_scores,
        relevance_labels=relevance,
        group_ids=group_ids,
        kl_target_strategy=kl_target_strategy,
        alpha_fair=alpha_fair,
        delta_high=delta_high,
    )
    if epsilon_target is None:
        return None

    min_yap = torch.min(yap)
    max_yap = torch.max(yap)
    yap_norm = (yap - min_yap) / torch.clamp(max_yap - min_yap, min=eps)
    relevance_weight = relevance if float(tau_relevance) == 1.0 else torch.pow(
        relevance.clamp_min(0.0),
        float(tau_relevance),
    )
    # beta_anti_yap controls the strength of the creator_anti_attention downweighting.
    anti_attention_weights = relevance_weight * torch.exp(-float(beta_anti_yap) * yap_norm)
    creator_probs = F.softmax(creator_scores, dim=-1)

    total_kl = creator_scores.new_zeros(())
    found_valid_group = False
    for group_idx in range(NUM_YAP_GROUPS):
        group_target_weight = epsilon_target[group_idx]
        if group_target_weight <= 0:
            continue

        group_membership = memberships[:, group_idx]
        target_weights = anti_attention_weights * group_membership
        pred_weights = creator_probs * group_membership
        target_sum = target_weights.sum()
        pred_sum = pred_weights.sum()
        if target_sum <= 0 or pred_sum <= 0:
            continue

        r_target = target_weights / target_sum.clamp_min(eps)
        r_pred = pred_weights / pred_sum.clamp_min(eps)
        valid_creators = r_target.gt(0)
        if not valid_creators.any():
            continue

        safe_target = r_target[valid_creators].clamp_min(eps)
        safe_pred = r_pred[valid_creators].clamp_min(eps)
        group_kl = torch.sum(safe_target * (torch.log(safe_target) - torch.log(safe_pred)))
        total_kl = total_kl + group_target_weight * group_kl
        found_valid_group = True

    if not found_valid_group:
        return None
    return total_kl


def _iter_stepwise_project_candidates(
    logits,
    labels,
    project_creator_token_ids,
    aligned_values=None,
    ignore_index=IGNORE_INDEX,
):
    shift_logits, shift_labels = _get_shifted_logits_and_labels(logits, labels)
    _, creator_step_positions, has_valid_target = get_creator_step_positions(
        labels,
        ignore_index=ignore_index,
    )

    for batch_idx, token_ids in enumerate(project_creator_token_ids):
        if not bool(has_valid_target[batch_idx]) or len(token_ids) == 0:
            continue

        sample_positions = creator_step_positions[batch_idx]
        if sample_positions.numel() == 0:
            continue

        token_ids = [int(token_id) for token_id in token_ids]
        token_to_index = {token_id: idx for idx, token_id in enumerate(token_ids)}

        sample_values = None
        if aligned_values is not None:
            sample_values = aligned_values[batch_idx]
            if len(sample_values) != len(token_ids):
                raise ValueError("Aligned per-project values must match project_creator_token_ids.")

        first_target_token_id = int(shift_labels[batch_idx, sample_positions[0]].item())
        if first_target_token_id not in token_to_index:
            continue

        already_generated = set()
        for step_position in sample_positions.tolist():
            target_token_id = int(shift_labels[batch_idx, step_position].item())
            if target_token_id not in token_to_index:
                break

            remaining_indices = [
                idx for idx, token_id in enumerate(token_ids) if token_id not in already_generated
            ]
            if not remaining_indices:
                break

            remaining_token_ids = [token_ids[idx] for idx in remaining_indices]
            remaining_logits = shift_logits[batch_idx, step_position].index_select(
                0,
                torch.tensor(
                    remaining_token_ids,
                    dtype=torch.long,
                    device=shift_logits.device,
                ),
            )

            yield {
                "remaining_values": None if sample_values is None else [sample_values[idx] for idx in remaining_indices],
                "remaining_logits": remaining_logits,
            }
            already_generated.add(target_token_id)


def compute_project_kl_loss(
    logits,
    labels,
    project_creator_token_ids,
    relevance_labels,
    group_ids,
    kl_target_strategy,
    alpha_fair,
    delta_high,
    ignore_index=IGNORE_INDEX,
):
    """Compute stepwise group-level KL over the current project's creator tokens only."""
    losses = []
    aligned_values = [
        list(zip(sample_relevance, sample_group_ids))
        for sample_relevance, sample_group_ids in zip(relevance_labels, group_ids)
    ]
    for step in _iter_stepwise_project_candidates(
        logits=logits,
        labels=labels,
        project_creator_token_ids=project_creator_token_ids,
        aligned_values=aligned_values,
        ignore_index=ignore_index,
    ):
        if not step["remaining_values"]:
            continue
        step_relevance, step_group_ids = zip(*step["remaining_values"])
        step_loss = _compute_group_kl_from_scores(
            creator_scores=step["remaining_logits"],
            relevance_labels=step_relevance,
            group_ids=step_group_ids,
            kl_target_strategy=kl_target_strategy,
            alpha_fair=alpha_fair,
            delta_high=delta_high,
        )
        if step_loss is not None:
            losses.append(step_loss)

    if not losses:
        return logits.new_zeros(())
    return torch.stack(losses).mean()


def compute_project_creator_kl_loss(
    logits,
    labels,
    project_creator_token_ids,
    relevance_labels,
    yap_scores,
    group_ids,
    creator_kl_strategy,
    kl_target_strategy,
    alpha_fair,
    delta_high,
    beta_anti_yap,
    tau_relevance,
    ignore_index=IGNORE_INDEX,
):
    """Compute creator-level anti-attention KL used only for SuppGroup+Creator."""
    creator_kl_strategy = normalize_creator_kl_strategy(creator_kl_strategy)
    if creator_kl_strategy == "none":
        return logits.new_zeros(())

    losses = []
    aligned_values = [
        list(zip(sample_relevance, sample_yap_scores, sample_group_ids))
        for sample_relevance, sample_yap_scores, sample_group_ids in zip(
            relevance_labels,
            yap_scores,
            group_ids,
        )
    ]
    for step in _iter_stepwise_project_candidates(
        logits=logits,
        labels=labels,
        project_creator_token_ids=project_creator_token_ids,
        aligned_values=aligned_values,
        ignore_index=ignore_index,
    ):
        if not step["remaining_values"]:
            continue
        step_relevance, step_yap_scores, step_group_ids = zip(*step["remaining_values"])
        step_loss = _compute_creator_anti_attention_kl_from_scores(
            creator_scores=step["remaining_logits"],
            relevance_labels=step_relevance,
            yap_scores=step_yap_scores,
            group_ids=step_group_ids,
            kl_target_strategy=kl_target_strategy,
            alpha_fair=alpha_fair,
            delta_high=delta_high,
            beta_anti_yap=beta_anti_yap,
            tau_relevance=tau_relevance,
        )
        if step_loss is not None:
            losses.append(step_loss)

    if not losses:
        return logits.new_zeros(())
    return torch.stack(losses).mean()


def compute_total_loss(
    *,
    logits,
    labels,
    all_creator_token_ids,
    project_creator_token_ids,
    rank_labels,
    project_yap_scores,
    project_group_ids,
    loss_type="ntp",
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
    eos_token_id=None,
    ignore_index=IGNORE_INDEX,
):
    validate_loss_type(loss_type)
    aux_loss_step_mode = normalize_aux_loss_step_mode(loss_type, aux_loss_step_mode)
    kl_target_strategy = normalize_kl_target_strategy(loss_type, kl_target_strategy)
    creator_kl_strategy = normalize_creator_kl_strategy(creator_kl_strategy)

    if eos_token_id is not None:
        all_creator_token_id_set = {int(token_id) for token_id in all_creator_token_ids.tolist()}
        if int(eos_token_id) not in all_creator_token_id_set:
            raise ValueError("all_creator_token_ids must include EOS.")
        for batch_idx, token_ids in enumerate(project_creator_token_ids):
            if int(eos_token_id) in {int(token_id) for token_id in token_ids}:
                raise ValueError(
                    "project_creator_token_ids must not include EOS. "
                    f"Found EOS in batch index {batch_idx}."
                )

    ntp_loss = compute_restricted_ntp_loss(
        logits=logits,
        labels=labels,
        allowed_token_ids=all_creator_token_ids,
        ignore_index=ignore_index,
    )

    zero = logits.new_zeros(())
    kl_loss = zero
    creator_kl_loss = zero
    total_loss = lambda_ntp * ntp_loss

    if loss_type == "ntp_kl":
        if aux_loss_step_mode != "stepwise":
            raise ValueError("Paper KL losses require aux_loss_step_mode='stepwise'.")
        kl_loss = compute_project_kl_loss(
            logits=logits,
            labels=labels,
            project_creator_token_ids=project_creator_token_ids,
            relevance_labels=rank_labels,
            group_ids=project_group_ids,
            kl_target_strategy=kl_target_strategy,
            alpha_fair=alpha_fair,
            delta_high=delta_high,
            ignore_index=ignore_index,
        )
        total_loss = total_loss + lambda_kl * kl_loss

        creator_kl_active = (
            float(lambda_kl_creator) != 0.0
            and creator_kl_strategy == "creator_anti_attention"
        )
        if creator_kl_active:
            creator_kl_loss = compute_project_creator_kl_loss(
                logits=logits,
                labels=labels,
                project_creator_token_ids=project_creator_token_ids,
                relevance_labels=rank_labels,
                yap_scores=project_yap_scores,
                group_ids=project_group_ids,
                creator_kl_strategy=creator_kl_strategy,
                kl_target_strategy=kl_target_strategy,
                alpha_fair=alpha_fair,
                delta_high=delta_high,
                beta_anti_yap=beta_anti_yap,
                tau_relevance=tau_relevance,
                ignore_index=ignore_index,
            )
            total_loss = total_loss + lambda_kl_creator * creator_kl_loss

    return total_loss, {
        "loss": total_loss.detach(),
        "ntp_loss": ntp_loss.detach(),
        "kl_loss": kl_loss.detach(),
        "creator_kl_loss": creator_kl_loss.detach(),
    }


class TeamFormationLoss:
    def __init__(
        self,
        mode="ntp",
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
        eos_token_id=None,
        ignore_index=IGNORE_INDEX,
    ):
        self.mode = validate_loss_type(mode)
        self.lambda_ntp = lambda_ntp
        self.lambda_kl = lambda_kl
        self.lambda_kl_creator = float(lambda_kl_creator)
        self.aux_loss_step_mode = normalize_aux_loss_step_mode(self.mode, aux_loss_step_mode)
        self.kl_target_strategy = normalize_kl_target_strategy(self.mode, kl_target_strategy)
        self.creator_kl_strategy = normalize_creator_kl_strategy(creator_kl_strategy)
        self.alpha_fair = float(alpha_fair)
        self.delta_high = float(delta_high)
        self.beta_anti_yap = float(beta_anti_yap)
        self.tau_relevance = float(tau_relevance)
        self.eos_token_id = eos_token_id
        self.ignore_index = ignore_index

    def __call__(
        self,
        *,
        logits,
        labels,
        all_creator_token_ids,
        project_creator_token_ids,
        rank_labels,
        project_yap_scores,
        project_group_ids,
    ):
        total_loss, loss_dict = compute_total_loss(
            logits=logits,
            labels=labels,
            all_creator_token_ids=all_creator_token_ids,
            project_creator_token_ids=project_creator_token_ids,
            rank_labels=rank_labels,
            project_yap_scores=project_yap_scores,
            project_group_ids=project_group_ids,
            loss_type=self.mode,
            lambda_ntp=self.lambda_ntp,
            lambda_kl=self.lambda_kl,
            lambda_kl_creator=self.lambda_kl_creator,
            aux_loss_step_mode=self.aux_loss_step_mode,
            kl_target_strategy=self.kl_target_strategy,
            creator_kl_strategy=self.creator_kl_strategy,
            alpha_fair=self.alpha_fair,
            delta_high=self.delta_high,
            beta_anti_yap=self.beta_anti_yap,
            tau_relevance=self.tau_relevance,
            eos_token_id=self.eos_token_id,
            ignore_index=self.ignore_index,
        )
        out = {
            "loss": total_loss,
            "ntp_loss": loss_dict["ntp_loss"],
            "kl_loss": loss_dict["kl_loss"],
        }
        if (
            self.lambda_kl_creator != 0.0
            and self.creator_kl_strategy == "creator_anti_attention"
        ):
            out["creator_kl_loss"] = loss_dict["creator_kl_loss"]
        return out
