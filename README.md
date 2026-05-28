# Exposure-Aware Generative Creator Ranking

This repository supports a Web3 creator-ranking task where a causal language model generates an ordered list of creator IDs for a given project prompt.

---

## Task

Given a project prompt and a project-specific candidate creator set, the model generates a ranked creator sequence.

Each training example contains:

- `prompt`: the project instruction shown to the model
- `target_text`: ordered creator IDs to generate
- `creators`: candidate creators for the project
- `label`: creator relevance label
- `rank`: creator rank when available
- `yap_score`: creator attention score
- `yap_bucket`: low, mid, or high attention group

---

## Paper Methods

The repository supports four paper methods.

| Paper method | `method_name` | `loss_type` | `kl_target_strategy` | `creator_kl_strategy` |
| --- | --- | --- | --- | --- |
| NTP | `ntp` | `ntp` | `none` | `none` |
| FairGroup | `fairgroup` | `ntp_kl` | `fair_group` | `none` |
| SuppGroup | `suppgroup` | `ntp_kl` | `high_suppressed_group` | `none` |
| SuppGroup+Creator | `suppgroup_creator` | `ntp_kl` | `high_suppressed_group` | `creator_anti_attention` |

The final creator-level method is configured as:

```text
SuppGroup+Creator
= method_name: suppgroup_creator
= loss_type: ntp_kl
= kl_target_strategy: high_suppressed_group
= creator_kl_strategy: creator_anti_attention
```

`creator_anti_attention` is the creator-level KL strategy used by SuppGroup+Creator. It downweights high-attention creators within each group while preserving relevance weighting.

The raw dataset field is still called `yap_score`, which represents the creator attention score.

---

## Losses

The active loss types are:

```text
ntp
ntp_kl
```

The total objective is:

```text
L = lambda_ntp * L_NTP
  + lambda_kl * L_group_KL
  + lambda_kl_creator * L_creator_KL
```

For `ntp`, only the NTP loss is active.

For `fairgroup` and `suppgroup`, the model uses NTP plus group-level KL.

For `suppgroup_creator`, the model uses NTP, group-level KL, and creator-level KL.

---

## Training

Install dependencies first:

```bash
pip install -r requirements.txt
```

Train the NTP baseline:

```bash
python train/reranker/train_reranker.py \
  --config train/reranker/config/paper_ntp.yaml
```

Train FairGroup:

```bash
python train/reranker/train_reranker.py \
  --config train/reranker/config/paper_fairgroup.yaml
```

Train SuppGroup:

```bash
python train/reranker/train_reranker.py \
  --config train/reranker/config/paper_suppgroup.yaml
```

Train SuppGroup+Creator:

```bash
python train/reranker/train_reranker.py \
  --config train/reranker/config/paper_suppgroup_creator.yaml
```

Training outputs are saved under:

```text
output/<dataset>/run/<timestamp>/
```

Each run saves:

```text
run_config.json
resolved_config.yaml
source_config.yaml
model/
logs/
```

---

## Evaluation

Evaluate a saved model:

```bash
python infer/evaluate_reranker.py \
  --jsonl data/kaito/test.jsonl \
  --model output/kaito/run/<timestamp>/model \
  --output_dir output/kaito/run/<timestamp> \
  --results_csv_name results.csv \
  --ks 10,20 \
  --max_new_tokens 32 \
  --batch_size 1
```


---

## Main Metrics

The main paper metrics are:

| Metric | Meaning |
| --- | --- |
| `graded_ndcg_at_10` | Ranking quality at 10 |
| `graded_ndcg_at_20` | Ranking quality at 20 |
| `avg_yaps_at_10` | Average attention-score exposure at 10 |
| `avg_yaps_at_20` | Average attention-score exposure at 20 |
| `disc_yaps_at_10` | Discounted attention-score exposure at 10 |
| `disc_yaps_at_20` | Discounted attention-score exposure at 20 |
| `exp_share_low/mid/high_at_10` | Raw exposure share by attention group |
| `dexp_share_low/mid/high_at_10` | Discounted exposure share by attention group |
| `dgroup_gap_at_10` | Distance from the target group exposure distribution |
| `within_group_gini_low/mid/high_at_10` | Exposure concentration within each group |

Every exposure-aware method should be compared against the plain `ntp` baseline for the same model.

---

## Dependencies

Install dependencies with:

```bash
pip install -r requirements.txt
```
