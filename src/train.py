from __future__ import annotations

import json
import math
import os
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import config_schema
import data as data_utils
import eval as eval_utils


_TRAINING_STACK: tuple[Any, ...] | None = None


def configure_hf_hub_runtime() -> dict[str, str]:
    settings = {
        "HF_HUB_DISABLE_XET": "1",
        "HF_HUB_DOWNLOAD_TIMEOUT": "60",
        "HF_HUB_ETAG_TIMEOUT": "30",
    }
    for key, value in settings.items():
        os.environ.setdefault(key, value)
    return {key: os.environ[key] for key in settings}


def _require_training_stack():
    global _TRAINING_STACK
    if _TRAINING_STACK is not None:
        return _TRAINING_STACK
    try:
        import torch  # type: ignore
        from peft import LoraConfig, TaskType, get_peft_model  # type: ignore
        from transformers import AutoModelForCausalLM, AutoTokenizer, get_scheduler  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "transformers, peft, and torch are required in the notebook kernel."
        ) from exc
    _TRAINING_STACK = (
        torch,
        AutoModelForCausalLM,
        AutoTokenizer,
        LoraConfig,
        TaskType,
        get_peft_model,
        get_scheduler,
    )
    return _TRAINING_STACK


def _autocast_context(torch: Any, precision: str, device_type: str):
    if device_type != "cuda":
        return nullcontext()
    if precision == "bfloat16":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    if precision == "float16":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def _append_jsonl(path: str | Path, row: dict[str, Any]) -> None:
    path = Path(path)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def resolve_runtime() -> dict[str, Any]:
    torch, *_ = _require_training_stack()
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        if torch.cuda.is_bf16_supported():
            return {
                "device": torch.device("cuda"),
                "device_type": "cuda",
                "precision": "bfloat16",
                "load_dtype": torch.bfloat16,
            }
        return {
            "device": torch.device("cuda"),
            "device_type": "cuda",
            "precision": "float16",
            "load_dtype": torch.float16,
        }
    return {
        "device": torch.device("cpu"),
        "device_type": "cpu",
        "precision": "float32",
        "load_dtype": None,
    }


def load_model_and_tokenizer(
    config: dict[str, Any],
    for_training: bool,
) -> tuple[Any, Any, dict[str, Any]]:
    configure_hf_hub_runtime()
    (
        _,
        AutoModelForCausalLM,
        AutoTokenizer,
        _,
        _,
        _,
        _,
    ) = _require_training_stack()
    runtime = resolve_runtime()
    trust_remote_code = config["model"]["trust_remote_code"]
    model_candidates = [config["model"]["name"]]
    if config["model"].get("fallback_name"):
        model_candidates.append(config["model"]["fallback_name"])

    last_error: Exception | None = None
    for model_name in model_candidates:
        try:
            print(f"[load] tokenizer start: {model_name}")
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=trust_remote_code,
            )
            print(f"[load] tokenizer done: {model_name}")
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token

            load_kwargs = {
                "trust_remote_code": trust_remote_code,
                "low_cpu_mem_usage": True,
            }
            if runtime["load_dtype"] is not None:
                load_kwargs["dtype"] = runtime["load_dtype"]
            try:
                print(f"[load] model start: {model_name}")
                model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
            except TypeError as exc:
                if runtime["load_dtype"] is None:
                    raise
                fallback_kwargs = dict(load_kwargs)
                fallback_kwargs.pop("dtype", None)
                fallback_kwargs["torch_dtype"] = runtime["load_dtype"]
                print(f"[load] model retry with torch_dtype: {model_name}")
                model = AutoModelForCausalLM.from_pretrained(model_name, **fallback_kwargs)
            print(f"[load] model done: {model_name}")

            if getattr(model.config, "pad_token_id", None) is None:
                model.config.pad_token_id = tokenizer.pad_token_id
            if for_training:
                if config["training"]["gradient_checkpointing"]:
                    model.gradient_checkpointing_enable()
                if hasattr(model.config, "use_cache"):
                    model.config.use_cache = False
            model.to(runtime["device"])
            runtime["effective_model_name"] = model_name
            return model, tokenizer, runtime
        except Exception as exc:  # pragma: no cover - runtime dependent
            last_error = exc

    raise RuntimeError(
        "Failed to load the configured base model and fallback model."
    ) from last_error


def load_model_from_checkpoint(
    config: dict[str, Any],
    checkpoint_path: str | Path | None = None,
) -> tuple[Any, Any, dict[str, Any]]:
    model, tokenizer, runtime_info = load_model_and_tokenizer(config, for_training=False)
    if config["experiment"]["run_type"] != "adapter":
        model.eval()
        return model, tokenizer, runtime_info

    checkpoint_path = Path(checkpoint_path or "")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Adapter checkpoint was not found: {checkpoint_path}")

    try:
        from peft import PeftModel  # type: ignore
    except ImportError as exc:
        raise ImportError("peft is required in the notebook kernel to reload adapter checkpoints.") from exc

    print(f"[load] adapter checkpoint start: {checkpoint_path}")
    model = PeftModel.from_pretrained(model, checkpoint_path)
    model.to(runtime_info["device"])
    model.eval()
    print(f"[load] adapter checkpoint done: {checkpoint_path}")
    return model, tokenizer, runtime_info


def get_num_hidden_layers(model: Any) -> int:
    if hasattr(model.config, "num_hidden_layers"):
        return int(model.config.num_hidden_layers)
    if hasattr(model.config, "text_config") and hasattr(model.config.text_config, "num_hidden_layers"):
        return int(model.config.text_config.num_hidden_layers)
    raise AttributeError("Could not infer num_hidden_layers from the model config.")


def count_trainable_parameters(model: Any) -> dict[str, int]:
    trainable = 0
    total = 0
    for parameter in model.parameters():
        count = parameter.numel()
        total += count
        if parameter.requires_grad:
            trainable += count
    return {"trainable": trainable, "total": total}


class CausalLMCollator:
    def __init__(self, tokenizer: Any):
        torch, *_ = _require_training_stack()
        self.torch = torch
        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.pad_token_id

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        max_len = max(len(feature["input_ids"]) for feature in features)
        input_ids = []
        attention_masks = []
        labels = []
        for feature in features:
            pad_length = max_len - len(feature["input_ids"])
            input_ids.append(feature["input_ids"] + [self.pad_token_id] * pad_length)
            attention_masks.append(feature["attention_mask"] + [0] * pad_length)
            labels.append(feature["labels"] + [-100] * pad_length)
        return {
            "input_ids": self.torch.tensor(input_ids, dtype=self.torch.long),
            "attention_mask": self.torch.tensor(attention_masks, dtype=self.torch.long),
            "labels": self.torch.tensor(labels, dtype=self.torch.long),
        }


def _move_batch_to_device(batch: dict[str, Any], device: Any) -> dict[str, Any]:
    non_blocking = getattr(device, "type", None) == "cuda"
    return {
        key: value.to(device, non_blocking=non_blocking)
        for key, value in batch.items()
    }


def _evaluate_token_loss(
    model: Any,
    data_loader: Any,
    runtime_info: dict[str, Any],
) -> float:
    torch, *_ = _require_training_stack()
    device = runtime_info["device"]
    losses: list[float] = []
    model.eval()
    with torch.inference_mode():
        for batch in data_loader:
            batch = _move_batch_to_device(batch, device)
            with _autocast_context(torch, runtime_info["precision"], runtime_info["device_type"]):
                outputs = model(**batch)
            losses.append(float(outputs.loss.item()))
    return sum(losses) / max(1, len(losses))


def _build_lora_model(model: Any, config: dict[str, Any]) -> tuple[Any, list[int]]:
    (
        _,
        _,
        _,
        LoraConfig,
        TaskType,
        get_peft_model,
        _,
    ) = _require_training_stack()
    layer_indices = config_schema.resolve_layer_indices(
        get_num_hidden_layers(model),
        config["experiment"]["placement"],
    )
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=int(config["experiment"]["rank"]),
        lora_alpha=int(config["model"]["lora_alpha"]),
        lora_dropout=float(config["model"]["lora_dropout"]),
        bias=config["model"]["bias"],
        target_modules=list(config["model"]["target_modules"]),
        layers_to_transform=layer_indices,
        layers_pattern=config["model"]["layer_pattern"],
    )
    return get_peft_model(model, lora_config), layer_indices


def train_adapter(
    config: dict[str, Any],
    run_dir: str | Path,
) -> tuple[Any, Any, dict[str, Any], dict[str, Any], dict[str, Any]]:
    (
        torch,
        _,
        _,
        _,
        _,
        _,
        get_scheduler,
    ) = _require_training_stack()
    run_dir = Path(run_dir)
    model, tokenizer, runtime_info = load_model_and_tokenizer(config, for_training=True)
    model, layer_indices = _build_lora_model(model, config)
    parameter_counts = count_trainable_parameters(model)

    domain_splits = data_utils.load_domain_splits(config)
    tokenized_train = data_utils.prepare_supervised_records(domain_splits["train"], tokenizer, config)
    tokenized_validation = data_utils.prepare_supervised_records(domain_splits["validation"], tokenizer, config)

    if not tokenized_train:
        raise ValueError("No train samples remained after tokenization or truncation.")
    if not tokenized_validation:
        raise ValueError("No validation samples remained after tokenization or truncation.")

    from torch.utils.data import DataLoader  # type: ignore

    train_loader = DataLoader(
        tokenized_train,
        batch_size=config["training"]["per_device_batch_size"],
        shuffle=True,
        collate_fn=CausalLMCollator(tokenizer),
        pin_memory=runtime_info["device_type"] == "cuda",
    )
    validation_loader = DataLoader(
        tokenized_validation,
        batch_size=config["training"]["per_device_eval_batch_size"],
        shuffle=False,
        collate_fn=CausalLMCollator(tokenizer),
        pin_memory=runtime_info["device_type"] == "cuda",
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    epochs = int(config["training"]["epochs"])
    gradient_accumulation_steps = int(config["training"]["gradient_accumulation_steps"])
    total_update_steps = max(
        1,
        math.ceil(len(train_loader) / gradient_accumulation_steps) * epochs,
    )
    scheduler = get_scheduler(
        name=config["training"]["scheduler_type"],
        optimizer=optimizer,
        num_warmup_steps=int(total_update_steps * float(config["training"]["warmup_ratio"])),
        num_training_steps=total_update_steps,
    )
    scaler = torch.cuda.amp.GradScaler(
        enabled=runtime_info["device_type"] == "cuda" and runtime_info["precision"] == "float16"
    )

    train_log_path = run_dir / "train_log.jsonl"
    start_time = time.time()
    global_step = 0
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        epoch_loss_total = 0.0
        epoch_loss_steps = 0
        for step, batch in enumerate(train_loader, start=1):
            batch = _move_batch_to_device(batch, runtime_info["device"])
            with _autocast_context(torch, runtime_info["precision"], runtime_info["device_type"]):
                outputs = model(**batch)
                raw_loss = outputs.loss
                loss = raw_loss / gradient_accumulation_steps

            if scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if step % gradient_accumulation_steps == 0 or step == len(train_loader):
                if scaler.is_enabled():
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

            epoch_loss_total += float(raw_loss.item())
            epoch_loss_steps += 1

        train_loss = epoch_loss_total / max(1, epoch_loss_steps)
        val_loss = _evaluate_token_loss(model, validation_loader, runtime_info)
        validation_output_path = None
        if config["training"].get("save_validation_predictions", False):
            validation_output_path = run_dir / f"predictions_domain_validation_epoch{epoch}.jsonl"
        val_metrics = eval_utils.evaluate_domain_generation(
            model=model,
            tokenizer=tokenizer,
            config=config,
            examples=domain_splits["validation"],
            output_path=validation_output_path,
            runtime_info=runtime_info,
            split_name=f"validation_epoch_{epoch}",
        )
        learning_rate = float(scheduler.get_last_lr()[0])
        log_row = {
            "epoch": epoch,
            "global_step": global_step,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_acc": val_metrics["f1"],
            "lr": learning_rate,
        }
        _append_jsonl(train_log_path, log_row)
        print(
            f"epoch={epoch} "
            f"train loss={train_loss:.4f} "
            f"val loss={val_loss:.4f} "
            f"val acc={val_metrics['f1']:.4f} "
            f"lr={learning_rate:.6f}"
        )

    checkpoint_dir = run_dir / "checkpoint-final"
    if config["training"]["save_checkpoint"]:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(checkpoint_dir)
        if config["training"].get("save_tokenizer_with_checkpoint", False):
            tokenizer.save_pretrained(checkpoint_dir)

    training_info = {
        "runtime_seconds": time.time() - start_time,
        "trainable_parameters": parameter_counts["trainable"],
        "total_parameters": parameter_counts["total"],
        "layer_indices": layer_indices,
        "checkpoint_path": str(checkpoint_dir),
    }
    return model, tokenizer, training_info, domain_splits, runtime_info


def cleanup_model(model: Any) -> None:
    torch, *_ = _require_training_stack()
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
