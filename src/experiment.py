from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from typing import Any, Mapping

import analysis as analysis_utils
import config_schema
import data as data_utils
import eval as eval_utils
import train as train_utils


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def prepare_config(
    base_config: str | Path | Mapping[str, Any] | None = None,
    overrides: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return config_schema.prepare_config(base_config=base_config, overrides=overrides)


def freeze_domain_dataset(
    config_or_path: str | Path | Mapping[str, Any] | None = None,
    overrides: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    config = prepare_config(config_or_path or "configs/pilot.yaml", overrides=overrides)
    _ensure_output_dirs(config)
    snapshot_info = data_utils.materialize_domain_snapshot(config)
    print(
        "[data] domain snapshot ready "
        f"(train={snapshot_info['split_sizes']['train']}, "
        f"validation={snapshot_info['split_sizes']['validation']}, "
        f"test={snapshot_info['split_sizes']['test']})"
    )
    return snapshot_info


def build_smoke_overrides(force_rerun: bool = False) -> dict[str, Any]:
    return {
        "experiment": {
            "run_name": "smoke_upper_r4_seed42",
            "placement": "upper",
            "rank": 4,
            "force_rerun": force_rerun,
        },
        "model": {
            "name": "Qwen/Qwen2.5-0.5B-Instruct",
            "fallback_name": None,
        },
        "data": {
            "domain": {
                "max_train_samples": 8,
                "max_validation_samples": 8,
                "max_test_samples": 8,
            },
            "general": {
                "boolq": {"max_samples": 8},
                "piqa": {"max_samples": 8},
                "wikitext": {"max_samples": 8, "max_segments": 8},
            },
        },
        "training": {
            "epochs": 1,
            "gradient_accumulation_steps": 1,
        },
    }


def build_mini_pilot_overrides(force_rerun: bool = False) -> dict[str, Any]:
    return {
        "experiment": {
            "run_name": "mini_upper_r4_seed42",
            "placement": "upper",
            "rank": 4,
            "force_rerun": force_rerun,
        },
        "data": {
            "domain": {
                "max_train_samples": 64,
                "max_validation_samples": 32,
                "max_test_samples": 32,
            },
            "general": {
                "boolq": {"max_samples": 64},
                "piqa": {"max_samples": 64},
                "wikitext": {"max_samples": 64, "max_segments": 32},
            },
        },
        "training": {
            "epochs": 1,
            "gradient_accumulation_steps": 8,
        },
    }


def probe_hf_access(
    config_or_path: str | Path | Mapping[str, Any] | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    config = prepare_config(config_or_path or "configs/pilot.yaml")
    train_utils.configure_hf_hub_runtime()
    try:
        from huggingface_hub import hf_hub_download  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "huggingface_hub is required in the notebook kernel to probe Hub access."
        ) from exc

    results = {"model": {}, "datasets": {}, "timings": {}}
    model_files = ["config.json", "tokenizer_config.json"]
    for filename in model_files:
        if verbose:
            print(f"[probe] model file start: {config['model']['name']} / {filename}")
        started_at = time.time()
        path = hf_hub_download(repo_id=config["model"]["name"], filename=filename)
        results["model"][filename] = path
        results["timings"][f"model::{filename}"] = round(time.time() - started_at, 3)
        if verbose:
            print(f"[probe] model file done: {filename} -> {path}")

    dataset_targets = {
        "domain": config["data"]["domain"]["path"],
        "boolq": config["data"]["general"]["boolq"]["path"],
        "piqa": config["data"]["general"]["piqa"]["path"],
        "wikitext": config["data"]["general"]["wikitext"]["path"],
    }
    for dataset_name, repo_id in dataset_targets.items():
        if verbose:
            print(f"[probe] dataset file start: {repo_id} / README.md")
        started_at = time.time()
        path = hf_hub_download(repo_id=repo_id, filename="README.md", repo_type="dataset")
        results["datasets"][dataset_name] = path
        results["timings"][f"dataset::{dataset_name}"] = round(time.time() - started_at, 3)
        if verbose:
            print(f"[probe] dataset file done: {dataset_name} -> {path}")

    return results


def diagnose_hf_access(timeout_seconds: int = 15) -> dict[str, Any]:
    train_utils.configure_hf_hub_runtime()
    results: dict[str, Any] = {
        "env": train_utils.configure_hf_hub_runtime(),
        "network": {},
        "libraries": {},
    }

    try:
        import huggingface_hub  # type: ignore

        results["libraries"]["huggingface_hub"] = getattr(huggingface_hub, "__version__", "unknown")
    except ImportError:
        results["libraries"]["huggingface_hub"] = "missing"

    try:
        import transformers  # type: ignore

        results["libraries"]["transformers"] = getattr(transformers, "__version__", "unknown")
    except ImportError:
        results["libraries"]["transformers"] = "missing"

    try:
        import requests  # type: ignore
    except ImportError as exc:
        raise ImportError("requests is required in the notebook kernel for diagnose_hf_access.") from exc

    urls = {
        "huggingface_home": "https://huggingface.co",
        "model_config": "https://huggingface.co/Qwen/Qwen2.5-3B-Instruct/resolve/main/config.json",
        "dataset_readme": "https://huggingface.co/datasets/google/boolq/resolve/main/README.md",
    }
    session = requests.Session()
    for name, url in urls.items():
        print(f"[diagnose] request start: {name} -> {url}")
        started_at = time.time()
        response = session.get(url, timeout=timeout_seconds, allow_redirects=True, stream=True)
        elapsed = round(time.time() - started_at, 3)
        results["network"][name] = {
            "status_code": response.status_code,
            "elapsed_seconds": elapsed,
            "final_url": response.url,
        }
        print(f"[diagnose] request done: {name} status={response.status_code} elapsed={elapsed}s")
        response.close()
    return results


def _ensure_output_dirs(config: Mapping[str, Any]) -> None:
    Path(config["outputs"]["root"]).mkdir(parents=True, exist_ok=True)
    Path(config["outputs"]["runs_dir"]).mkdir(parents=True, exist_ok=True)
    Path(config["outputs"]["summary_dir"]).mkdir(parents=True, exist_ok=True)


def _reset_run_dir(run_dir: Path) -> None:
    for pattern in (
        "metrics.json",
        "config_snapshot.yaml",
        "train_log.jsonl",
        "predictions_*.jsonl",
        "predictions_domain_validation_epoch*.jsonl",
    ):
        for path in run_dir.glob(pattern):
            if path.is_file():
                path.unlink()
    checkpoint_dir = run_dir / "checkpoint-final"
    if checkpoint_dir.exists():
        shutil.rmtree(checkpoint_dir)


def _load_matrix(matrix_config: str | Path | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(matrix_config, Mapping):
        return dict(matrix_config)
    return config_schema.load_config_file(matrix_config)


def _resolve_run_dir(config: Mapping[str, Any]) -> Path:
    return Path(config["outputs"]["runs_dir"]) / str(config["experiment"]["run_name"])


def _build_metrics_payload(
    config: dict[str, Any],
    domain_splits: dict[str, Any],
    domain_metrics: dict[str, Any],
    general_metrics: dict[str, Any],
    training_info: dict[str, Any],
    runtime_info: dict[str, Any] | None,
) -> dict[str, Any]:
    return {
        "experiment": config["experiment"],
        "model": {
            "configured_name": config["model"]["name"],
            "effective_name": runtime_info["effective_model_name"] if runtime_info else config["model"]["name"],
            "target_modules": config["model"]["target_modules"],
        },
        "data": {
            "domain_name": data_utils.resolve_domain_dataset_name(config),
            "domain_dataset": config["data"]["domain"]["path"],
            "domain_input_mode": data_utils.infer_domain_input_mode(domain_splits["train"]),
            "domain_context_summary": data_utils.summarize_domain_context(domain_splits["train"]),
            "domain_context_fetch": dict(domain_splits.get("context_meta", {})),
            "boolq_dataset": config["data"]["general"]["boolq"]["path"],
            "piqa_dataset": config["data"]["general"]["piqa"]["path"],
            "wikitext_dataset": config["data"]["general"]["wikitext"]["path"],
        },
        "domain": domain_metrics,
        "general": general_metrics,
        "training": training_info,
    }


def _load_run_config_snapshot(
    run_name: str,
    outputs_root: str | Path = "outputs",
) -> tuple[dict[str, Any], Path]:
    outputs_root = Path(outputs_root)
    run_dir = outputs_root / "runs" / run_name
    config_path = run_dir / "config_snapshot.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Run config snapshot was not found: {config_path}")
    return prepare_config(config_path), run_dir


def reevaluate_run(
    run_name: str,
    outputs_root: str | Path = "outputs",
) -> dict[str, Any]:
    config, run_dir = _load_run_config_snapshot(run_name, outputs_root=outputs_root)
    _ensure_output_dirs(config)
    metrics_path = run_dir / "metrics.json"
    existing_metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else {}

    checkpoint_path: Path | None = None
    if config["experiment"]["run_type"] == "adapter":
        raw_checkpoint_path = str(existing_metrics.get("training", {}).get("checkpoint_path", "")).strip()
        if raw_checkpoint_path:
            candidate = Path(raw_checkpoint_path)
            checkpoint_path = candidate if candidate.exists() else None
        if checkpoint_path is None:
            checkpoint_path = run_dir / "checkpoint-final"

    print(f"[eval] start: {run_name}")
    model = None
    runtime_info: dict[str, Any] | None = None
    try:
        model, tokenizer, runtime_info = train_utils.load_model_from_checkpoint(
            config,
            checkpoint_path=checkpoint_path if config["experiment"]["run_type"] == "adapter" else None,
        )
        domain_splits = data_utils.load_domain_splits(config)
        domain_metrics = eval_utils.evaluate_domain_generation(
            model=model,
            tokenizer=tokenizer,
            config=config,
            examples=domain_splits["test"],
            output_path=run_dir / "predictions_domain.jsonl",
            runtime_info=runtime_info,
            split_name="test",
        )
        general_metrics = eval_utils.evaluate_general_benchmarks(
            model=model,
            tokenizer=tokenizer,
            config=config,
            runtime_info=runtime_info,
            run_dir=run_dir,
        )

        if existing_metrics.get("training"):
            training_info = dict(existing_metrics["training"])
        else:
            parameter_counts = train_utils.count_trainable_parameters(model)
            training_info = {
                "runtime_seconds": 0.0,
                "trainable_parameters": parameter_counts["trainable"],
                "total_parameters": parameter_counts["total"],
                "layer_indices": [],
                "checkpoint_path": str(checkpoint_path) if checkpoint_path is not None else "",
            }

        metrics = _build_metrics_payload(
            config=config,
            domain_splits=domain_splits,
            domain_metrics=domain_metrics,
            general_metrics=general_metrics,
            training_info=training_info,
            runtime_info=runtime_info,
        )
        _write_json(metrics_path, metrics)
        print(f"[eval] done: {run_name}")
        return metrics
    finally:
        if model is not None:
            train_utils.cleanup_model(model)


def reevaluate_batch(
    run_names: list[str] | None = None,
    outputs_root: str | Path = "outputs",
    include_aux_runs: bool = False,
) -> dict[str, Any]:
    outputs_root = Path(outputs_root)
    runs_dir = outputs_root / "runs"
    if run_names is None:
        run_names = sorted(path.name for path in runs_dir.iterdir() if path.is_dir())

    results = []
    total_runs = len(run_names)
    for index, run_name in enumerate(run_names, start=1):
        print(f"[eval-batch] {index}/{total_runs}: {run_name}")
        results.append(reevaluate_run(run_name, outputs_root=outputs_root))

    summary = analysis_utils.summarize_results(outputs_root, include_aux_runs=include_aux_runs)
    return {
        "runs": results,
        "summary": summary,
    }


def run_experiment(
    config_or_path: str | Path | Mapping[str, Any] | None = None,
    overrides: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    config = prepare_config(config_or_path or "configs/pilot.yaml", overrides=overrides)
    _ensure_output_dirs(config)

    run_dir = _resolve_run_dir(config)
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / "metrics.json"
    if metrics_path.exists() and not config["experiment"]["force_rerun"]:
        print(f"[run] skip existing metrics: {config['experiment']['run_name']}")
        return json.loads(metrics_path.read_text(encoding="utf-8"))
    if config["experiment"]["force_rerun"]:
        print(f"[run] reset artifacts: {config['experiment']['run_name']}")
        _reset_run_dir(run_dir)

    print(f"[run] start: {config['experiment']['run_name']}")
    config_schema.write_config_snapshot(run_dir / "config_snapshot.yaml", config)

    model = None
    tokenizer = None
    runtime_info: dict[str, Any] | None = None
    try:
        if config["experiment"]["run_type"] == "base":
            model, tokenizer, runtime_info = train_utils.load_model_and_tokenizer(config, for_training=False)
            domain_splits = data_utils.load_domain_splits(config)
            domain_metrics = eval_utils.evaluate_domain_generation(
                model=model,
                tokenizer=tokenizer,
                config=config,
                examples=domain_splits["test"],
                output_path=run_dir / "predictions_domain.jsonl",
                runtime_info=runtime_info,
                split_name="test",
            )
            general_metrics = eval_utils.evaluate_general_benchmarks(
                model=model,
                tokenizer=tokenizer,
                config=config,
                runtime_info=runtime_info,
                run_dir=run_dir,
            )
            parameter_counts = train_utils.count_trainable_parameters(model)
            training_info = {
                "runtime_seconds": 0.0,
                "trainable_parameters": 0,
                "total_parameters": parameter_counts["total"],
                "layer_indices": [],
                "checkpoint_path": "",
            }
        else:
            model, tokenizer, training_info, domain_splits, runtime_info = train_utils.train_adapter(
                config=config,
                run_dir=run_dir,
            )
            domain_metrics = eval_utils.evaluate_domain_generation(
                model=model,
                tokenizer=tokenizer,
                config=config,
                examples=domain_splits["test"],
                output_path=run_dir / "predictions_domain.jsonl",
                runtime_info=runtime_info,
                split_name="test",
            )
            general_metrics = eval_utils.evaluate_general_benchmarks(
                model=model,
                tokenizer=tokenizer,
                config=config,
                runtime_info=runtime_info,
                run_dir=run_dir,
            )

        metrics = _build_metrics_payload(
            config=config,
            domain_splits=domain_splits,
            domain_metrics=domain_metrics,
            general_metrics=general_metrics,
            training_info=training_info,
            runtime_info=runtime_info,
        )
        _write_json(metrics_path, metrics)
        print(f"[run] done: {config['experiment']['run_name']}")
        return metrics
    finally:
        if model is not None:
            train_utils.cleanup_model(model)


def run_batch(
    matrix_config_or_path: str | Path | Mapping[str, Any] = "configs/matrix/pilot_rank_placement.yaml",
    force_rerun: bool = False,
    include_aux_runs: bool = False,
) -> dict[str, Any]:
    matrix_config = _load_matrix(matrix_config_or_path)
    base_config_path = matrix_config.get("base_config_path", "configs/pilot.yaml")
    base_config_raw = config_schema.load_config_file(base_config_path)
    base_config = prepare_config(base_config_path)
    run_configs = config_schema.build_batch_configs(base_config_raw, matrix_config)
    placement_order = {"lower": 0, "middle": 1, "upper": 2, "all": 3}
    run_configs.sort(
        key=lambda cfg: (
            cfg["data"]["domain"].get("name", ""),
            cfg["experiment"]["run_type"] != "base",
            placement_order.get(str(cfg["experiment"]["placement"]), 99),
            int(cfg["experiment"]["rank"]),
            str(cfg["experiment"]["run_name"]),
        )
    )

    results = []
    total_runs = len(run_configs)
    for index, run_config in enumerate(run_configs, start=1):
        if force_rerun:
            run_config = config_schema.prepare_config(
                base_config=run_config,
                overrides={"experiment": {"force_rerun": True}},
            )
        print(f"[batch] {index}/{total_runs}: {run_config['experiment']['run_name']}")
        results.append(run_experiment(run_config))

    summary = analysis_utils.summarize_results(
        base_config["outputs"]["root"],
        include_aux_runs=include_aux_runs,
    )
    return {
        "runs": results,
        "summary": summary,
    }


def export_results_bundle(
    outputs_root: str | Path = "outputs",
    results_root: str | Path = "results",
    include_aux_runs: bool = False,
) -> dict[str, Any]:
    return analysis_utils.export_results_bundle(
        outputs_root=outputs_root,
        results_root=results_root,
        include_aux_runs=include_aux_runs,
    )
