import json
from ast import literal_eval
from collections import defaultdict
from pathlib import Path

import torch
import tqdm
from torch.utils.data import Dataset


def _safe_parse_creator_list(raw_value):
    if raw_value is None:
        return []
    text = str(raw_value).strip()
    if not text:
        return []
    try:
        parsed = literal_eval(text)
    except Exception:
        parsed = None

    if isinstance(parsed, (list, tuple, set)):
        return [str(value).strip() for value in parsed if str(value).strip()]
    if parsed is not None:
        parsed_text = str(parsed).strip()
        return [parsed_text] if parsed_text else []

    text = text.strip("[]")
    pieces = [piece.strip().strip("'").strip('"') for piece in text.split(",")]
    return [piece for piece in pieces if piece]


def collect_creator_ids_from_jsonl(
    data_paths,
    creators_key="creators",
    creator_id_key="creator_id",
    team_key="team_creator_ids",
):
    creator_ids = set()
    for data_path in data_paths:
        if not data_path:
            continue
        path = Path(data_path)
        if not path.exists():
            continue
        with open(path, encoding="utf-8") as f:
            for line in f:
                data_dic = json.loads(line.strip())
                for creator in data_dic.get(creators_key, []):
                    creator_id = creator.get(creator_id_key)
                    if creator_id is not None:
                        creator_ids.add(str(creator_id))
                for creator_id in data_dic.get(team_key, []) or []:
                    creator_ids.add(str(creator_id))
    return sorted(creator_id for creator_id in creator_ids if creator_id)


def collect_creator_ids_from_creators_csv(creators_csv_path):
    creator_ids = set()
    path = Path(creators_csv_path)
    if not path.exists():
        return []

    if path.suffix.lower() == ".json":
        with open(path, encoding="utf-8") as f:
            rows = json.load(f)
        for row in rows:
            for creator_id in _safe_parse_creator_list(row.get("creators")):
                creator_ids.add(str(creator_id))
        return sorted(creator_id for creator_id in creator_ids if creator_id)

    import csv

    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for creator_id in _safe_parse_creator_list(row.get("creators")):
                creator_ids.add(str(creator_id))
    return sorted(creator_id for creator_id in creator_ids if creator_id)


class TeamFormationDataset(Dataset):
    PROMPT_TEMPLATE = """Task: Generate the highest-ranked relevant creators for the project.

Project name: {project_name}
Project description: {project_description}
Required skills:
{skill_descriptions}

Return only creator IDs in order.
Do not repeat any creator.
Do not output explanations.
"""

    def __init__(
        self,
        data_path=None,
        target_model=None,
        max_len=2048,
        tag="train",
        task_type="team_formation",
        creator_id_key="creator_id",
        creator_yap_key="yap_score",
        project_name_key="project_name",
        project_description_key="project_description",
        required_skills_key="required_skills",
        creators_key="creators",
        team_key="team_creator_ids",
        creator_label_key="label",
        creator_rank_key="rank",
    ):
        assert data_path is not None and target_model is not None
        if task_type != "team_formation":
            raise ValueError(f"Unsupported task_type: {task_type}")

        self.model = target_model
        self.max_len = max_len
        self.tag = tag
        self.task_type = task_type
        self.creator_id_key = creator_id_key
        self.creator_yap_key = creator_yap_key
        self.project_name_key = project_name_key
        self.project_description_key = project_description_key
        self.required_skills_key = required_skills_key
        self.creators_key = creators_key
        self.team_key = team_key
        self.creator_label_key = creator_label_key
        self.creator_rank_key = creator_rank_key

        self.data = self.read_data(data_path)

    def _normalize_target_creator_ids(self, target_text, creators):
        creator_token_map = {
            str(creator.get("creator_token")): str(creator[self.creator_id_key])
            for creator in creators
            if creator.get("creator_token") is not None
        }
        normalized = []
        for token in str(target_text).strip().split():
            token = str(token).strip()
            if not token:
                continue
            normalized.append(creator_token_map.get(token, token))
        return normalized

    def _format_skills(self, required_skills):
        if isinstance(required_skills, str):
            return required_skills.strip()

        formatted_skills = []
        for skill in required_skills:
            if isinstance(skill, str):
                formatted_skills.append(skill.strip())
            elif isinstance(skill, dict):
                name = skill.get("skill") or skill.get("name") or skill.get("title") or ""
                description = skill.get("description") or skill.get("desc") or ""
                if name and description:
                    formatted_skills.append(f"{name}: {description}")
                else:
                    formatted_skills.append(f"{name or description}".rstrip())
            else:
                formatted_skills.append(f"{str(skill).strip()}")
        return "\n".join(formatted_skills)

    def _build_rank_labels(self, creators, team_creator_ids):
        explicit_labels = [creator.get(self.creator_label_key) for creator in creators]
        if all(label is not None for label in explicit_labels):
            return [float(label) for label in explicit_labels]

        explicit_ranks = [creator.get(self.creator_rank_key) for creator in creators]
        if all(rank is not None for rank in explicit_ranks):
            max_rank = max(int(rank) for rank in explicit_ranks)
            return [float(max_rank - int(rank) + 1) for rank in explicit_ranks]

        team_score_map = {
            creator_id: float(len(team_creator_ids) - idx)
            for idx, creator_id in enumerate(team_creator_ids)
        }
        return [
            team_score_map.get(str(creator[self.creator_id_key]), 0.0)
            for creator in creators
        ]

    def _build_project_yap_scores(self, creators):
        return [float(creator.get(self.creator_yap_key, 0.0)) for creator in creators]

    def _build_project_group_ids(self, creators, data_dic):
        bucket_to_id = {"low": 0, "mid": 1, "high": 2}
        explicit_buckets = [creator.get("yap_bucket") for creator in creators]
        if all(bucket in bucket_to_id for bucket in explicit_buckets):
            return [bucket_to_id[bucket] for bucket in explicit_buckets]

        group_map = {}
        for bucket_name, creator_ids in (data_dic.get("yap_bucket_groups") or {}).items():
            if bucket_name in bucket_to_id:
                for creator_id in creator_ids or []:
                    group_map[str(creator_id)] = bucket_to_id[bucket_name]
        if len(group_map) == len(creators):
            return [group_map[str(creator[self.creator_id_key])] for creator in creators]

        scored_creators = sorted(
            enumerate(creators),
            key=lambda item: (
                float(item[1].get(self.creator_yap_key, 0.0)),
                str(item[1][self.creator_id_key]),
            ),
        )
        n = len(scored_creators)
        base = n // 3
        remainder = n % 3
        sizes = [
            base + (1 if remainder > 0 else 0),
            base + (1 if remainder > 1 else 0),
            base,
        ]
        derived_group_ids = [0] * n
        start = 0
        for group_id, size in enumerate(sizes):
            end = start + size
            for original_idx, _ in scored_creators[start:end]:
                derived_group_ids[original_idx] = group_id
            start = end
        return derived_group_ids

    def read_data(self, data_path):
        data = []
        team_size_distribution = defaultdict(int)
        with open(data_path, encoding="utf-8") as f:
            for line in tqdm.tqdm(f):
                data_dic = json.loads(line.strip())
                creators = data_dic[self.creators_key]
                if len(creators) == 0:
                    continue
                if len(creators) > self.model.max_creator_candidates:
                    raise ValueError(
                        f"Found {len(creators)} creators, but max_creator_candidates="
                        f"{self.model.max_creator_candidates}. Increase the config value."
                    )

                project_id = str(
                    data_dic.get("project_id")
                    or data_dic.get(self.project_name_key)
                    or data_dic.get("target_type")
                    or data_dic.get("id")
                    or ""
                ).strip()
                project_name = str(data_dic.get(self.project_name_key, "")).strip()
                project_description = str(data_dic.get(self.project_description_key, "")).strip()
                required_skills = self._format_skills(data_dic.get(self.required_skills_key, []))

                prompt = str(data_dic.get("prompt", "")).strip()
                if not prompt:
                    prompt = self.PROMPT_TEMPLATE.format(
                        project_name=project_name,
                        project_description=project_description,
                        skill_descriptions=required_skills,
                        creator_metadata="",
                    ).strip()

                project_creator_ids = [str(creator[self.creator_id_key]) for creator in creators]
                project_creator_token_ids = self.model.get_project_creator_token_ids(project_creator_ids)

                target_text = str(data_dic.get("target_text", "")).strip()
                team_creator_ids = data_dic.get(self.team_key) or data_dic.get("team") or []
                team_creator_ids = [str(creator_id) for creator_id in team_creator_ids]
                target_creator_ids = self._normalize_target_creator_ids(target_text, creators)
                if not target_creator_ids:
                    target_creator_ids = list(team_creator_ids)
                if len(target_creator_ids) == 0:
                    continue
                target_creator_token_ids = self.model.get_project_creator_token_ids(target_creator_ids)

                rank_labels = self._build_rank_labels(creators, team_creator_ids or target_creator_ids)
                project_yap_scores = self._build_project_yap_scores(creators)
                project_group_ids = self._build_project_group_ids(creators, data_dic)

                data.append(
                    {
                        "prompt": prompt,
                        "target_text": target_text,
                        "project_id": project_id or project_name,
                        "target_creator_ids": target_creator_ids,
                        "target_creator_token_ids": target_creator_token_ids,
                        "project_creator_ids": project_creator_ids,
                        "project_creator_token_ids": project_creator_token_ids,
                        "project_rank_labels": rank_labels,
                        "project_yap_scores": project_yap_scores,
                        "project_group_ids": project_group_ids,
                    }
                )
                team_size_distribution[str(len(target_creator_ids))] += 1

        if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
            print(f"----- {self.tag} data -----")
            print(f"Loaded {len(data)} team formation examples")
            print(f"Team size distribution: {dict(team_size_distribution)}")

        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def collate_fn(self, batch):
        return self.model.preprocess(batch, self.max_len)
