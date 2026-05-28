# Team-Formation Reranker

This directory contains the active LLM decoder training path for generative
creator ranking. The old generic pointwise, pairwise, and grouped reranker
dataset interfaces have been removed from the research release.

## Active Dataset Format

Training expects team-formation JSONL records with:

- `prompt`: instruction shown to the model.
- `target_text`: ordered creator IDs to generate.
- `creators`: project candidate creators with `creator_id`, `label`, `rank`,
  `yap_score`, and optional `yap_bucket`.

Build the JSONL files with:

```bash
python train/reranker/build_jsonl.py
```

The generated JSONL files are expected directly under `data/kaito/`, such as
`train.jsonl`, `valid.jsonl`, and `test.jsonl`.

## Paper Methods

Use the paper config files under `train/reranker/config/`:

- `paper_ntp.yaml`: plain next-token prediction.
- `paper_fairgroup.yaml`: NTP plus fair group KL.
- `paper_suppgroup.yaml`: NTP plus high-group suppression KL.
- `paper_suppgroup_creator.yaml`: SuppGroup plus creator-level anti-attention KL.

Run a method with:

```bash
python train/reranker/train_reranker.py --config train/reranker/config/paper_ntp.yaml
```

See the top-level `README.md` for the paper-method to internal-config mapping.

`creator_anti_attention` is the creator-level KL strategy used by
SuppGroup+Creator. It downweights high-attention creators within each group
while preserving relevance weighting. The raw dataset field is still called
`yap_score`, which represents the creator attention score.
