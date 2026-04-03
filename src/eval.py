from __future__ import annotations

from collections import Counter
import json
import math
import re
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Iterable

import data as data_utils


_TORCH_MODULE: Any | None = None


def _require_torch():
    global _TORCH_MODULE
    if _TORCH_MODULE is not None:
        return _TORCH_MODULE
    try:
        import torch  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "PyTorch is required in the notebook kernel to run evaluation."
        ) from exc
    _TORCH_MODULE = torch
    return _TORCH_MODULE


def _move_to_device(batch: dict[str, Any], device: Any) -> dict[str, Any]:
    non_blocking = getattr(device, "type", None) == "cuda"
    return {
        key: value.to(device, non_blocking=non_blocking)
        for key, value in batch.items()
    }


def _batched(items: list[Any], batch_size: int) -> Iterable[list[Any]]:
    for index in range(0, len(items), batch_size):
        yield items[index : index + batch_size]


def _write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    path = Path(path)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize_answer(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return " ".join(text.split())


def exact_match_score(prediction: str, reference: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(reference))


def f1_score(prediction: str, reference: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    ref_tokens = normalize_answer(reference).split()
    if not pred_tokens and not ref_tokens:
        return 1.0
    if not pred_tokens or not ref_tokens:
        return 0.0

    overlap = sum((Counter(pred_tokens) & Counter(ref_tokens)).values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def _generation_context(torch: Any, precision: str, device_type: str):
    if device_type != "cuda":
        return nullcontext()
    if precision == "bfloat16":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    if precision == "float16":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def generate_texts(
    model: Any,
    tokenizer: Any,
    prompts: list[str],
    config: dict[str, Any],
    runtime_info: dict[str, Any],
) -> list[str]:
    torch = _require_torch()
    batch_size = config["evaluation"]["generation_batch_size"]
    max_new_tokens = config["evaluation"]["generation_max_new_tokens"]
    device = runtime_info["device"]
    precision = runtime_info["precision"]

    original_padding_side = getattr(tokenizer, "padding_side", "right")
    tokenizer.padding_side = "left"
    outputs: list[str] = []
    try:
        for prompt_batch in _batched(prompts, batch_size):
            encoded = tokenizer(
                prompt_batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=config["model"]["max_seq_len"],
            )
            encoded = _move_to_device(encoded, device)
            with torch.inference_mode():
                with _generation_context(torch, precision, runtime_info["device_type"]):
                    generated = model.generate(
                        **encoded,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                        top_p=config["evaluation"]["top_p"],
                        temperature=config["evaluation"]["temperature"],
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                    )
            attention_mask = encoded["attention_mask"]
            for row_index in range(generated.size(0)):
                prompt_length = int(attention_mask[row_index].sum().item())
                generated_ids = generated[row_index][prompt_length:]
                outputs.append(tokenizer.decode(generated_ids, skip_special_tokens=True).strip())
    finally:
        tokenizer.padding_side = original_padding_side

    return outputs


def evaluate_domain_generation(
    model: Any,
    tokenizer: Any,
    config: dict[str, Any],
    examples: list[dict[str, Any]],
    output_path: str | Path | None,
    runtime_info: dict[str, Any],
    split_name: str,
) -> dict[str, Any]:
    prepared = data_utils.prepare_domain_generation_records(examples, tokenizer, config)
    predictions = generate_texts(
        model=model,
        tokenizer=tokenizer,
        prompts=[row["prompt_text"] for row in prepared],
        config=config,
        runtime_info=runtime_info,
    )

    rows = []
    f1_scores: list[float] = []
    for prepared_row, prediction in zip(prepared, predictions):
        em = exact_match_score(prediction, prepared_row["reference_answer"])
        f1 = f1_score(prediction, prepared_row["reference_answer"])
        f1_scores.append(f1)
        rows.append(
            {
                "id": prepared_row["id"],
                "split": split_name,
                "filename": prepared_row.get("filename", ""),
                "context_source": prepared_row.get("context_source", ""),
                "context_url": prepared_row.get("context_url", ""),
                "question": prepared_row["question"],
                "context": prepared_row["context"],
                "reference_answer": prepared_row["reference_answer"],
                "prediction": prediction,
                "exact_match": em,
                "f1": f1,
            }
        )

    if output_path is not None:
        _write_jsonl(output_path, rows)
    return {
        "samples": len(f1_scores),
        "f1": sum(f1_scores) / max(1, len(f1_scores)),
    }


def _parse_boolq_prediction(text: str) -> str:
    normalized = normalize_answer(text)
    if normalized.startswith("yes"):
        return "yes"
    if normalized.startswith("no"):
        return "no"
    if " yes " in f" {normalized} ":
        return "yes"
    if " no " in f" {normalized} ":
        return "no"
    return normalized.split()[0] if normalized else ""


def evaluate_boolq(
    model: Any,
    tokenizer: Any,
    config: dict[str, Any],
    runtime_info: dict[str, Any],
    output_path: str | Path,
) -> dict[str, Any]:
    examples = data_utils.load_boolq_examples(config)
    prompts = [
        data_utils.apply_chat_template(
            tokenizer,
            data_utils.build_chat_messages(
                config["prompt"]["system"],
                config["prompt"]["boolq_template"].format(
                    passage=example["passage"],
                    question=example["question"],
                ),
            ),
            add_generation_prompt=True,
        )
        for example in examples
    ]
    predictions = generate_texts(model, tokenizer, prompts, config, runtime_info)

    rows = []
    correct = 0
    for example, prediction in zip(examples, predictions):
        parsed_prediction = _parse_boolq_prediction(prediction)
        is_correct = int(parsed_prediction == example["label"])
        correct += is_correct
        rows.append(
            {
                "question": example["question"],
                "reference": example["label"],
                "prediction": prediction,
                "parsed_prediction": parsed_prediction,
                "correct": is_correct,
            }
        )
    _write_jsonl(output_path, rows)
    return {
        "samples": len(rows),
        "accuracy": correct / max(1, len(rows)),
    }


def _parse_piqa_prediction(text: str) -> str:
    stripped = text.strip().upper()
    if stripped.startswith("A") or stripped.startswith("1"):
        return "A"
    if stripped.startswith("B") or stripped.startswith("2"):
        return "B"
    match = re.search(r"\b(A|B|1|2)\b", stripped)
    if not match:
        return ""
    return "A" if match.group(1) in {"A", "1"} else "B"


def evaluate_piqa(
    model: Any,
    tokenizer: Any,
    config: dict[str, Any],
    runtime_info: dict[str, Any],
    output_path: str | Path,
) -> dict[str, Any]:
    examples = data_utils.load_piqa_examples(config)
    prompts = [
        data_utils.apply_chat_template(
            tokenizer,
            data_utils.build_chat_messages(
                config["prompt"]["system"],
                config["prompt"]["piqa_template"].format(
                    goal=example["goal"],
                    sol1=example["sol1"],
                    sol2=example["sol2"],
                ),
            ),
            add_generation_prompt=True,
        )
        for example in examples
    ]
    predictions = generate_texts(model, tokenizer, prompts, config, runtime_info)

    rows = []
    correct = 0
    for example, prediction in zip(examples, predictions):
        parsed_prediction = _parse_piqa_prediction(prediction)
        is_correct = int(parsed_prediction == example["label"])
        correct += is_correct
        rows.append(
            {
                "goal": example["goal"],
                "reference": example["label"],
                "prediction": prediction,
                "parsed_prediction": parsed_prediction,
                "correct": is_correct,
            }
        )
    _write_jsonl(output_path, rows)
    return {
        "samples": len(rows),
        "accuracy": correct / max(1, len(rows)),
    }


def evaluate_wikitext_perplexity(
    model: Any,
    tokenizer: Any,
    config: dict[str, Any],
    runtime_info: dict[str, Any],
) -> dict[str, Any]:
    torch = _require_torch()
    texts = data_utils.load_wikitext_texts(config)
    if not texts:
        return {"samples": 0, "perplexity": math.inf}

    joined_text = "\n\n".join(texts)
    encoded = tokenizer(joined_text, return_tensors="pt")
    input_ids = encoded["input_ids"][0]
    max_length = config["model"]["max_seq_len"]
    max_segments = config["data"]["general"]["wikitext"].get("max_segments")
    device = runtime_info["device"]

    total_loss = 0.0
    total_tokens = 0
    segment_count = 0
    model.eval()
    for start in range(0, input_ids.size(0), max_length):
        if max_segments is not None and segment_count >= max_segments:
            break
        end = min(start + max_length, input_ids.size(0))
        chunk = input_ids[start:end]
        if chunk.numel() <= 1:
            continue
        batch = {
            "input_ids": chunk.unsqueeze(0).to(device),
            "attention_mask": torch.ones((1, chunk.size(0)), dtype=torch.long, device=device),
            "labels": chunk.unsqueeze(0).to(device),
        }
        with torch.inference_mode():
            with _generation_context(torch, runtime_info["precision"], runtime_info["device_type"]):
                loss = model(**batch).loss
        token_count = chunk.size(0)
        total_loss += float(loss.item()) * token_count
        total_tokens += token_count
        segment_count += 1

    average_loss = total_loss / max(1, total_tokens)
    return {
        "samples": segment_count,
        "perplexity": math.exp(average_loss),
    }


def evaluate_general_benchmarks(
    model: Any,
    tokenizer: Any,
    config: dict[str, Any],
    runtime_info: dict[str, Any],
    run_dir: str | Path,
) -> dict[str, Any]:
    run_dir = Path(run_dir)
    boolq = evaluate_boolq(
        model=model,
        tokenizer=tokenizer,
        config=config,
        runtime_info=runtime_info,
        output_path=run_dir / "predictions_boolq.jsonl",
    )
    piqa = evaluate_piqa(
        model=model,
        tokenizer=tokenizer,
        config=config,
        runtime_info=runtime_info,
        output_path=run_dir / "predictions_piqa.jsonl",
    )
    wikitext = evaluate_wikitext_perplexity(
        model=model,
        tokenizer=tokenizer,
        config=config,
        runtime_info=runtime_info,
    )
    return {
        "boolq": boolq,
        "piqa": piqa,
        "wikitext": wikitext,
    }
