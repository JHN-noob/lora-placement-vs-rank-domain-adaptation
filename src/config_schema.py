from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Mapping


DEFAULT_PILOT_CONFIG: dict[str, Any] = {
    "experiment": {
        "name": "pilot_rank_placement",
        "seed": 42,
        "run_type": "adapter",
        "placement": "upper",
        "rank": 4,
        "run_name": None,
        "force_rerun": False,
    },
    "model": {
        "name": "Qwen/Qwen2.5-3B-Instruct",
        "fallback_name": "meta-llama/Llama-3.2-3B-Instruct",
        "target_modules": ["q_proj", "v_proj"],
        "layer_pattern": "layers",
        "lora_dropout": 0.05,
        "lora_alpha": None,
        "bias": "none",
        "trust_remote_code": False,
        "max_seq_len": 768,
    },
    "data": {
        "domain": {
            "name": "keboola_docs",
            "path": "Keboola/Developer-Documentation-QA",
            "config_name": None,
            "question_field": None,
            "context_field": None,
            "answer_field": None,
            "group_by_field": "filename",
            "split_seed": 42,
            "split_ratio": [0.8, 0.1, 0.1],
            "context_strategy": "keboola_docs_by_filename",
            "context_base_url": "https://developers.keboola.com",
            "context_cache_path": "outputs/cache/keboola_context_cache.json",
            "context_cache_version": "v2_h1_article_body",
            "prepared_snapshot_path": "outputs/cache/keboola_domain_snapshot.json",
            "prepared_snapshot_version": "v2_context_clean",
            "prefer_prepared_snapshot": True,
            "context_fetch_timeout_seconds": 30,
            "context_fetch_workers": 8,
            "context_max_chars": 5000,
            "context_fit_max_tokens": 384,
            "answer_token_reserve": 128,
            "min_context_chars": 80,
            "min_context_body_lines": 1,
            "max_train_samples": None,
            "max_validation_samples": None,
            "max_test_samples": None,
        },
        "general": {
            "boolq": {
                "path": "google/boolq",
                "config_name": None,
                "split": "validation",
                "max_samples": None,
            },
            "piqa": {
                "path": "ybisk/piqa",
                "config_name": None,
                "split": "validation",
                "max_samples": None,
            },
            "wikitext": {
                "path": "Salesforce/wikitext",
                "config_name": "wikitext-2-raw-v1",
                "split": "validation",
                "max_samples": None,
                "max_segments": None,
            },
        },
    },
    "prompt": {
        "system": (
            "You are a technical documentation assistant. "
            "Answer technical questions accurately and concisely."
        ),
        "domain_user_template": (
            "Documentation source: {filename}\n\n"
            "Documentation:\n{context}\n\n"
            "Question: {question}\n\n"
            "Answer as accurately as possible."
        ),
        "domain_question_only_template": (
            "Documentation source: {filename}\n\n"
            "Question: {question}\n\n"
            "Answer as accurately as possible."
        ),
        "boolq_template": (
            "Passage:\n{passage}\n\n"
            "Question: {question}\n\n"
            "Answer with only yes or no."
        ),
        "piqa_template": (
            "Goal: {goal}\n\n"
            "A. {sol1}\n"
            "B. {sol2}\n\n"
            "Which option is more reasonable? Answer with only A or B."
        ),
    },
    "training": {
        "epochs": 3,
        "per_device_batch_size": 1,
        "per_device_eval_batch_size": 1,
        "gradient_accumulation_steps": 16,
        "learning_rate": 2e-4,
        "scheduler_type": "cosine",
        "warmup_ratio": 0.03,
        "weight_decay": 0.0,
        "gradient_checkpointing": True,
        "save_checkpoint": True,
        "save_validation_predictions": False,
        "save_tokenizer_with_checkpoint": False,
    },
    "evaluation": {
        "generation_max_new_tokens": 96,
        "generation_batch_size": 1,
        "temperature": 0.0,
        "top_p": 1.0,
    },
    "outputs": {
        "root": "outputs",
        "runs_dir": "outputs/runs",
        "summary_dir": "outputs/summary",
    },
}

CANONICAL_BASE_RUN_NAMES = {
    "keboola_docs": "keboola_base",
    "techqa": "techqa_base",
}


def default_pilot_config() -> dict[str, Any]:
    return copy.deepcopy(DEFAULT_PILOT_CONFIG)


def load_config_file(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        try:
            import yaml  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                f"Could not parse config file {path}. "
                "Use JSON-compatible YAML or install PyYAML in the notebook kernel."
            ) from exc
        loaded = yaml.safe_load(text)
        if not isinstance(loaded, dict):
            raise ValueError(f"Config file must decode to a dict: {path}")
        return loaded


def write_config_snapshot(path: str | Path, config: Mapping[str, Any]) -> None:
    path = Path(path)
    try:
        import yaml  # type: ignore
    except ImportError:
        serialized = json.dumps(config, ensure_ascii=False, indent=2)
    else:
        serialized = yaml.safe_dump(
            json.loads(json.dumps(config, ensure_ascii=False)),
            allow_unicode=True,
            sort_keys=False,
        )
    path.write_text(serialized, encoding="utf-8")


def merge_dicts(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(dict(base))
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = merge_dicts(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def compute_layer_partitions(num_hidden_layers: int) -> dict[str, list[int]]:
    if num_hidden_layers <= 0:
        raise ValueError("num_hidden_layers must be positive.")

    base = num_hidden_layers // 3
    remainder = num_hidden_layers % 3
    sizes = [base, base, base]
    for index in range(remainder):
        sizes[index] += 1

    boundaries: list[tuple[int, int]] = []
    start = 0
    for size in sizes:
        end = start + size
        boundaries.append((start, end))
        start = end

    lower = list(range(*boundaries[0]))
    middle = list(range(*boundaries[1]))
    upper = list(range(*boundaries[2]))
    return {
        "lower": lower,
        "middle": middle,
        "upper": upper,
        "all": list(range(num_hidden_layers)),
    }


def resolve_layer_indices(num_hidden_layers: int, placement: str) -> list[int]:
    partitions = compute_layer_partitions(num_hidden_layers)
    if placement not in partitions:
        raise ValueError(f"Unsupported placement: {placement}")
    return partitions[placement]


def build_run_name(config: Mapping[str, Any]) -> str:
    experiment = config["experiment"]
    domain_name = str(config.get("data", {}).get("domain", {}).get("name", "")).strip()
    if experiment["run_type"] == "base":
        return CANONICAL_BASE_RUN_NAMES.get(domain_name, f"{domain_name or 'domain'}_base")

    base_name = (
        f"{experiment['placement']}_"
        f"r{experiment['rank']}_"
        f"seed{experiment['seed']}"
    )
    if domain_name and domain_name != "keboola_docs":
        return f"{domain_name}_{base_name}"
    return base_name


def validate_config(config: Mapping[str, Any]) -> dict[str, Any]:
    validated = merge_dicts(default_pilot_config(), config)

    experiment = validated["experiment"]
    if experiment["run_type"] not in {"base", "adapter"}:
        raise ValueError("experiment.run_type must be 'base' or 'adapter'.")

    if experiment["run_type"] == "adapter" and experiment["placement"] not in {
        "lower",
        "middle",
        "upper",
        "all",
    }:
        raise ValueError("experiment.placement must be one of lower/middle/upper/all.")

    if experiment["run_type"] == "adapter" and int(experiment["rank"]) <= 0:
        raise ValueError("experiment.rank must be positive.")

    split_ratio = validated["data"]["domain"]["split_ratio"]
    if len(split_ratio) != 3 or abs(sum(split_ratio) - 1.0) > 1e-6:
        raise ValueError("data.domain.split_ratio must have three values summing to 1.0.")

    context_strategy = validated["data"]["domain"].get("context_strategy")
    if context_strategy not in {None, "keboola_docs_by_filename"}:
        raise ValueError("data.domain.context_strategy must be null or 'keboola_docs_by_filename'.")
    if int(validated["data"]["domain"].get("context_fit_max_tokens", 0)) <= 0:
        raise ValueError("data.domain.context_fit_max_tokens must be positive.")
    if int(validated["data"]["domain"].get("answer_token_reserve", 0)) <= 0:
        raise ValueError("data.domain.answer_token_reserve must be positive.")
    if int(validated["data"]["domain"].get("min_context_chars", 0)) < 0:
        raise ValueError("data.domain.min_context_chars must be non-negative.")
    if int(validated["data"]["domain"].get("min_context_body_lines", 0)) < 0:
        raise ValueError("data.domain.min_context_body_lines must be non-negative.")
    if not str(validated["data"]["domain"].get("prepared_snapshot_version", "")).strip():
        raise ValueError("data.domain.prepared_snapshot_version must be non-empty.")
    if not str(validated["data"]["domain"].get("name", "")).strip():
        raise ValueError("data.domain.name must be non-empty.")
    if not isinstance(validated["training"].get("save_validation_predictions"), bool):
        raise ValueError("training.save_validation_predictions must be a boolean.")
    if not isinstance(validated["training"].get("save_tokenizer_with_checkpoint"), bool):
        raise ValueError("training.save_tokenizer_with_checkpoint must be a boolean.")

    outputs = validated["outputs"]
    outputs["root"] = str(Path(outputs["root"]))
    outputs["runs_dir"] = str(Path(outputs["runs_dir"]))
    outputs["summary_dir"] = str(Path(outputs["summary_dir"]))

    model = validated["model"]
    if model["lora_alpha"] is None:
        model["lora_alpha"] = experiment["rank"]

    if not experiment.get("run_name"):
        experiment["run_name"] = build_run_name(validated)

    return validated


def prepare_config(
    base_config: str | Path | Mapping[str, Any] | None = None,
    overrides: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if base_config is None:
        config = default_pilot_config()
    elif isinstance(base_config, Mapping):
        config = merge_dicts(default_pilot_config(), base_config)
    else:
        loaded = load_config_file(base_config)
        config = merge_dicts(default_pilot_config(), loaded)

    if overrides:
        config = merge_dicts(config, overrides)
        experiment_override = overrides.get("experiment", {}) if isinstance(overrides, Mapping) else {}
        model_override = overrides.get("model", {}) if isinstance(overrides, Mapping) else {}
        if "rank" in experiment_override and "lora_alpha" not in model_override:
            config["model"]["lora_alpha"] = None
    return validate_config(config)


def build_batch_configs(
    base_config: Mapping[str, Any],
    matrix_config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    runs = matrix_config.get("runs", [])
    configs: list[dict[str, Any]] = []
    for run_override in runs:
        configs.append(prepare_config(base_config=base_config, overrides=run_override))
    return configs
