from __future__ import annotations

import copy
import concurrent.futures
import html
import io
import json
import random
import re
import zipfile
from pathlib import Path
from typing import Any, Iterable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


QUESTION_FIELD_CANDIDATES = ("question", "query", "prompt", "instruction")
CONTEXT_FIELD_CANDIDATES = (
    "context",
    "passage",
    "document",
    "documentation",
    "doc",
    "reference",
    "content",
    "body",
    "text",
)
ANSWER_FIELD_CANDIDATES = (
    "answer",
    "answers",
    "response",
    "output",
    "completion",
    "label",
)


_DOMAIN_SPLITS_CACHE: dict[tuple[Any, ...], dict[str, Any]] = {}
_GENERAL_DATA_CACHE: dict[tuple[Any, ...], list[Any]] = {}


def _require_datasets():
    try:
        from datasets import concatenate_datasets, load_dataset  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "The datasets package is required in the notebook kernel to run this pipeline."
        ) from exc
    return load_dataset, concatenate_datasets


def _coerce_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (list, tuple)):
        for item in value:
            text = _coerce_text(item)
            if text:
                return text
        return ""
    if isinstance(value, dict):
        for key in ("text", "answer", "answers", "value", "output", "response", "content"):
            if key in value:
                return _coerce_text(value[key])
        return json.dumps(value, ensure_ascii=False)
    return str(value).strip()


def _pick_field(example: dict[str, Any], override: str | None, candidates: Iterable[str]) -> str | None:
    if override:
        return override
    for candidate in candidates:
        if candidate in example:
            return candidate
    return None


def _infer_domain_field_map(example: dict[str, Any], config: dict[str, Any]) -> dict[str, str | None]:
    domain_config = config["data"]["domain"]
    question_field = _pick_field(example, domain_config.get("question_field"), QUESTION_FIELD_CANDIDATES)
    answer_field = _pick_field(example, domain_config.get("answer_field"), ANSWER_FIELD_CANDIDATES)
    context_field = _pick_field(example, domain_config.get("context_field"), CONTEXT_FIELD_CANDIDATES)

    if not question_field or not answer_field:
        raise ValueError(
            "Could not infer question/answer fields from the domain dataset. "
            "Set data.domain.question_field and data.domain.answer_field explicitly."
        )

    return {
        "question": question_field,
        "answer": answer_field,
        "context": context_field,
        "filename": "filename" if "filename" in example else None,
    }


def _normalize_domain_example(
    example: dict[str, Any],
    field_map: dict[str, str | None],
    index: int,
) -> dict[str, Any]:
    question = _coerce_text(example.get(field_map["question"]))
    answer = _coerce_text(example.get(field_map["answer"]))
    context = _coerce_text(example.get(field_map["context"])) if field_map.get("context") else ""
    filename = _coerce_text(example.get(field_map["filename"])) if field_map.get("filename") else ""
    return {
        "id": str(example.get("id", index)),
        "question": question,
        "answer": answer,
        "context": context,
        "filename": filename,
        "context_source": "dataset" if context else "qa_only",
        "context_url": "",
    }


def _limit_records(records: list[dict[str, Any]], limit: int | None) -> list[dict[str, Any]]:
    if limit is None:
        return list(records)
    return records[: min(limit, len(records))]


def resolve_domain_dataset_name(config: dict[str, Any]) -> str:
    domain_config = config["data"]["domain"]
    explicit_name = _coerce_text(domain_config.get("name"))
    if explicit_name:
        return explicit_name
    dataset_path = _coerce_text(domain_config.get("path"))
    dataset_name = dataset_path.split("/")[-1] if dataset_path else "domain"
    return re.sub(r"[^a-z0-9]+", "_", dataset_name.lower()).strip("_") or "domain"


def _domain_cache_key(domain_config: dict[str, Any]) -> tuple[Any, ...]:
    return (
        domain_config.get("name"),
        domain_config["path"],
        domain_config.get("config_name"),
        domain_config.get("question_field"),
        domain_config.get("context_field"),
        domain_config.get("answer_field"),
        domain_config.get("group_by_field"),
        domain_config.get("context_strategy"),
        domain_config.get("context_base_url"),
        domain_config.get("context_cache_path"),
        domain_config.get("context_cache_version"),
        domain_config.get("context_max_chars"),
        domain_config.get("context_fit_max_tokens"),
        domain_config.get("answer_token_reserve"),
        domain_config.get("min_context_chars"),
        domain_config.get("min_context_body_lines"),
        domain_config.get("prepared_snapshot_path"),
        domain_config.get("prepared_snapshot_version"),
        domain_config.get("prefer_prepared_snapshot"),
        domain_config.get("max_train_samples"),
        domain_config.get("max_validation_samples"),
        domain_config.get("max_test_samples"),
        tuple(domain_config["split_ratio"]),
        domain_config["split_seed"],
    )


def _general_cache_key(dataset_config: dict[str, Any], prefix: str) -> tuple[Any, ...]:
    return (
        prefix,
        dataset_config["path"],
        dataset_config.get("config_name"),
        dataset_config["split"],
    )


def _reshuffle_splits(raw_splits: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    load_dataset, concatenate_datasets = _require_datasets()
    del load_dataset

    datasets_to_merge = [dataset for dataset in raw_splits.values()]
    if not datasets_to_merge:
        raise ValueError("Domain dataset did not contain any splits.")

    merged = concatenate_datasets(datasets_to_merge).shuffle(seed=config["data"]["domain"]["split_seed"])
    group_by_field = config["data"]["domain"].get("group_by_field")
    if group_by_field and group_by_field in merged.column_names:
        return _grouped_split_by_field(merged, config, group_by_field)

    split_ratio = config["data"]["domain"]["split_ratio"]
    train_end = int(len(merged) * split_ratio[0])
    validation_end = train_end + int(len(merged) * split_ratio[1])
    return {
        "train": merged.select(range(0, train_end)),
        "validation": merged.select(range(train_end, validation_end)),
        "test": merged.select(range(validation_end, len(merged))),
    }


def _grouped_split_by_field(merged: Any, config: dict[str, Any], group_by_field: str) -> dict[str, Any]:
    split_ratio = config["data"]["domain"]["split_ratio"]
    rng = random.Random(config["data"]["domain"]["split_seed"])
    grouped_indices: dict[str, list[int]] = {}
    for index in range(len(merged)):
        row = merged[index]
        group_value = _coerce_text(row.get(group_by_field))
        if not group_value:
            group_value = f"__missing__{index}"
        grouped_indices.setdefault(group_value, []).append(index)

    group_keys = list(grouped_indices)
    rng.shuffle(group_keys)

    total_rows = len(merged)
    train_target = int(total_rows * split_ratio[0])
    validation_target = int(total_rows * split_ratio[1])

    split_indices = {"train": [], "validation": [], "test": []}
    for group_key in group_keys:
        indices = grouped_indices[group_key]
        if len(split_indices["train"]) < train_target:
            target_split = "train"
        elif len(split_indices["validation"]) < validation_target:
            target_split = "validation"
        else:
            target_split = "test"
        split_indices[target_split].extend(indices)

    return {
        split_name: merged.select(sorted(indices))
        for split_name, indices in split_indices.items()
    }


def _context_cache_path(config: dict[str, Any]) -> str:
    configured = config["data"]["domain"].get("context_cache_path")
    if configured:
        return configured
    return str(Path(config["outputs"]["root"]) / "cache" / "keboola_context_cache.json")


def _context_cache_version(config: dict[str, Any]) -> str:
    return str(config["data"]["domain"].get("context_cache_version", "v1"))


def _prepared_snapshot_path(config: dict[str, Any]) -> str:
    configured = config["data"]["domain"].get("prepared_snapshot_path")
    if configured:
        return configured
    return str(Path(config["outputs"]["root"]) / "cache" / "domain_snapshot.json")


def _prepared_snapshot_version(config: dict[str, Any]) -> str:
    return str(config["data"]["domain"].get("prepared_snapshot_version", "v1"))


def _load_context_cache(path: str | Path) -> dict[str, Any]:
    cache_path = Path(path)
    if not cache_path.exists():
        return {}
    try:
        loaded = json.loads(cache_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return loaded if isinstance(loaded, dict) else {}


def _write_context_cache(path: str | Path, payload: dict[str, Any]) -> None:
    cache_path = Path(path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _load_json_dict(path: str | Path) -> dict[str, Any]:
    json_path = Path(path)
    if not json_path.exists():
        return {}
    try:
        loaded = json.loads(json_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return loaded if isinstance(loaded, dict) else {}


def _write_json_dict(path: str | Path, payload: dict[str, Any]) -> None:
    json_path = Path(path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _is_valid_context_cache_entry(entry: dict[str, Any], cache_version: str) -> bool:
    return (
        bool(entry)
        and entry.get("status") == "ok"
        and entry.get("cache_version") == cache_version
        and bool(_coerce_text(entry.get("context")))
    )


def _guess_context_title(text: str) -> str:
    for raw_line in text.splitlines():
        line = " ".join(raw_line.split())
        if line:
            return line
    return ""


def _keboola_doc_url_candidates(filename: str, base_url: str) -> list[str]:
    cleaned = filename.strip()
    if not cleaned:
        return []

    path = "/" + cleaned.lstrip("/")
    candidates: list[str] = []
    if path.endswith("/index.md"):
        candidates.append(f"{base_url}{path[:-len('index.md')]}")
    if path.endswith(".md"):
        stem = path[:-len(".md")]
        candidates.append(f"{base_url}{stem}/")
        candidates.append(f"{base_url}{stem}")

    deduped: list[str] = []
    for candidate in candidates:
        if candidate not in deduped:
            deduped.append(candidate)
    return deduped


def _strip_tags(text: str) -> str:
    return " ".join(html.unescape(re.sub(r"<[^>]+>", " ", text)).split())


def _extract_document_title(html_text: str) -> str:
    h1_match = re.search(r"<h1\b[^>]*>(.*?)</h1>", html_text, flags=re.IGNORECASE | re.DOTALL)
    if h1_match:
        title = _strip_tags(h1_match.group(1))
        if title:
            return title

    title_match = re.search(r"<title\b[^>]*>(.*?)</title>", html_text, flags=re.IGNORECASE | re.DOTALL)
    if title_match:
        title = _strip_tags(title_match.group(1))
        if "|" in title:
            return title.split("|", 1)[0].strip()
        return title
    return ""


def _extract_article_html(html_text: str) -> str:
    match = re.search(r"<main\b.*?</main>", html_text, flags=re.IGNORECASE | re.DOTALL)
    main_html = match.group(0) if match else html_text
    main_html = re.sub(r"<(script|style|svg|noscript|nav|aside|footer)\b.*?</\1>", " ", main_html, flags=re.IGNORECASE | re.DOTALL)
    h1_match = re.search(r"<h1\b[^>]*>.*?</h1>", main_html, flags=re.IGNORECASE | re.DOTALL)
    if h1_match:
        main_html = main_html[h1_match.start():]
    return main_html


def _looks_like_body_line(line: str) -> bool:
    if len(line) >= 80:
        return True
    if len(line.split()) >= 10 and any(token in line for token in (".", ":", ";", "?", "!")):
        return True
    return False


def _prune_to_article_lines(lines: list[str], title: str) -> list[str]:
    if not lines:
        return lines

    start_index = 0
    if title:
        matching_indices = [
            index for index, line in enumerate(lines)
            if line == title or line.startswith(f"{title} |")
        ]
        if matching_indices:
            start_index = matching_indices[-1]

    pruned = lines[start_index:]
    if not pruned:
        return lines

    lead_index = 1 if title and pruned[0] == title else 0
    while lead_index < len(pruned):
        if _looks_like_body_line(pruned[lead_index]):
            break
        lead_index += 1

    if title and pruned and pruned[0] == title:
        kept_lines = [pruned[0]] + pruned[lead_index:]
    else:
        kept_lines = pruned[lead_index:] if lead_index < len(pruned) else pruned
    return kept_lines or pruned


def _normalize_context_line(line: str) -> str:
    normalized = line
    normalized = normalized.replace("\u2019", "'").replace("\u2018", "'")
    normalized = normalized.replace("\u201c", '"').replace("\u201d", '"')
    normalized = normalized.replace("\u2013", "-").replace("\u2014", "-")
    normalized = re.sub(r"([A-Za-z])\?([A-Za-z])", r"\1'\2", normalized)
    normalized = re.sub(r"[^\x00-\x7F]+", "", normalized)
    normalized = normalized.replace("??", " ")
    normalized = re.sub(r"\?{2,}", " ", normalized)
    normalized = re.sub(r"(?<!\?)\?(?!\?)", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    normalized = re.sub(r"\s+([,.;:!?])", r"\1", normalized)
    return normalized


def _clean_context_text(context_text: str, title: str, domain_config: dict[str, Any]) -> str:
    footer_markers = (
        "2026 Keboola",
        "Keboola Developers Knowledge Base",
        "DEVELOPERS DOCS",
    )
    boilerplate_prefixes = (
        "Use Parameter Map",
        "To configure your first Generic Extractor",
    )
    stop_prefixes = (
        "Next Steps",
        "Global Options",
    )
    boilerplate_exact_lines = {
        "Copy",
        "Open page",
        "Edit page",
    }

    cleaned_lines: list[str] = []
    seen_lines: set[str] = set()
    for raw_line in context_text.splitlines():
        line = _normalize_context_line(raw_line)
        if not line:
            continue
        if line in {"-", "--"}:
            continue
        if line in boilerplate_exact_lines:
            continue
        if any(marker in line for marker in footer_markers):
            continue
        if any(line.startswith(prefix) for prefix in boilerplate_prefixes):
            continue
        if any(line.startswith(prefix) for prefix in stop_prefixes):
            break
        if len(line) <= 2:
            continue
        if line.lower().startswith(("was this page helpful", "on this page", "table of contents")):
            continue
        if line in seen_lines:
            continue
        seen_lines.add(line)
        cleaned_lines.append(line)

    while cleaned_lines and not cleaned_lines[-1]:
        cleaned_lines.pop()

    if not cleaned_lines:
        return ""

    if title:
        title = _normalize_context_line(title)

    if title and cleaned_lines[0] != title:
        cleaned_lines.insert(0, title)

    body_lines = [line for line in cleaned_lines if line != title]
    body_text = "\n".join(body_lines).strip()
    min_context_chars = int(domain_config.get("min_context_chars", 0))
    min_context_body_lines = int(domain_config.get("min_context_body_lines", 0))
    if len(body_text) < min_context_chars or len(body_lines) < min_context_body_lines:
        return ""

    return "\n".join(cleaned_lines)


def _sanitize_cache_entry(
    entry: dict[str, Any],
    cache_version: str,
    domain_config: dict[str, Any],
) -> tuple[dict[str, Any], bool]:
    if not isinstance(entry, dict):
        return {
            "status": "missing",
            "url": "",
            "context": "",
            "source": "qa_only",
            "cache_version": cache_version,
        }, True

    changed = False
    sanitized = dict(entry)
    if sanitized.get("cache_version") != cache_version:
        sanitized["cache_version"] = cache_version
        changed = True

    raw_context = _coerce_text(sanitized.get("context"))
    title = _guess_context_title(raw_context)
    cleaned_context = _clean_context_text(raw_context, title, domain_config)
    if cleaned_context:
        if cleaned_context != raw_context:
            sanitized["context"] = cleaned_context
            changed = True
        if sanitized.get("status") != "ok":
            sanitized["status"] = "ok"
            changed = True
        if sanitized.get("source") != "keboola_docs":
            sanitized["source"] = "keboola_docs"
            changed = True
        sanitized.setdefault("url", "")
        return sanitized, changed

    missing_entry = {
        "status": "missing",
        "url": _coerce_text(sanitized.get("url")),
        "context": "",
        "source": "qa_only",
        "cache_version": cache_version,
    }
    if sanitized != missing_entry:
        changed = True
    return missing_entry, changed


def _strip_html_to_text(html_text: str) -> str:
    title = _extract_document_title(html_text)
    main_html = _extract_article_html(html_text)
    main_html = re.sub(r"<(script|style|svg|noscript)\b.*?</\1>", " ", main_html, flags=re.IGNORECASE | re.DOTALL)
    main_html = re.sub(r"<li\b[^>]*>", "\n- ", main_html, flags=re.IGNORECASE)
    main_html = re.sub(r"</(p|div|section|article|main|h1|h2|h3|h4|h5|h6|pre|code|ul|ol|table|tr)>", "\n", main_html, flags=re.IGNORECASE)
    main_html = re.sub(r"<br\s*/?>", "\n", main_html, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", main_html)
    text = html.unescape(text)

    lines = []
    for raw_line in text.splitlines():
        line = " ".join(raw_line.split())
        if not line:
            continue
        if line.startswith(("Home", "CLI", "Commands")) and len(line.split()) <= 3:
            continue
        lowered = line.lower()
        if lowered.startswith("copyright") or "all rights reserved" in lowered:
            continue
        lines.append(line)
    return "\n".join(_prune_to_article_lines(lines, title))


def _truncate_context(text: str, max_chars: int | None) -> str:
    normalized = text.strip()
    if not normalized or max_chars is None or len(normalized) <= max_chars:
        return normalized
    return normalized[:max_chars].rsplit(" ", 1)[0].strip()


def _truncate_token_text(tokenizer: Any, text: str, max_tokens: int) -> str:
    if not text or max_tokens <= 0:
        return ""
    token_ids = tokenizer(
        text,
        add_special_tokens=False,
        truncation=True,
        max_length=max_tokens,
    )["input_ids"]
    if not token_ids:
        return ""
    return tokenizer.decode(token_ids, skip_special_tokens=True).strip()


def _count_text_tokens(tokenizer: Any, text: str) -> int:
    if not text:
        return 0
    return len(
        tokenizer(
            text,
            add_special_tokens=False,
            truncation=False,
        )["input_ids"]
    )


def _build_prompt_text(tokenizer: Any, config: dict[str, Any], example: dict[str, Any]) -> str:
    user_prompt = format_domain_user_prompt(config, example)
    return apply_chat_template(
        tokenizer,
        build_chat_messages(config["prompt"]["system"], user_prompt),
        add_generation_prompt=True,
    )


def _fit_context_to_prompt_budget(
    example: dict[str, Any],
    tokenizer: Any,
    config: dict[str, Any],
    prompt_budget: int,
) -> tuple[dict[str, Any], bool]:
    if prompt_budget <= 0 or not has_distinct_context(example):
        return dict(example), False

    context_limit = int(config["data"]["domain"]["context_fit_max_tokens"])
    candidate = dict(example)
    candidate["context"] = _truncate_token_text(tokenizer, example["context"], context_limit)
    if not candidate["context"]:
        return dict(example), False

    def prompt_length_for(context_text: str) -> int:
        candidate_example = dict(example)
        candidate_example["context"] = context_text
        return _count_text_tokens(tokenizer, _build_prompt_text(tokenizer, config, candidate_example))

    current_length = prompt_length_for(candidate["context"])
    if current_length <= prompt_budget:
        return candidate, candidate["context"] != example["context"]

    context_ids = tokenizer(
        candidate["context"],
        add_special_tokens=False,
        truncation=False,
    )["input_ids"]
    low = 1
    high = len(context_ids)
    best_context = ""
    while low <= high:
        mid = (low + high) // 2
        truncated_context = tokenizer.decode(context_ids[:mid], skip_special_tokens=True).strip()
        if not truncated_context:
            high = mid - 1
            continue
        current_length = prompt_length_for(truncated_context)
        if current_length <= prompt_budget:
            best_context = truncated_context
            low = mid + 1
        else:
            high = mid - 1

    if best_context:
        fitted = dict(example)
        fitted["context"] = best_context
        return fitted, True

    fallback = dict(example)
    fallback["context"] = ""
    fallback["context_source"] = "qa_only"
    fallback["context_url"] = ""
    return fallback, True


def _prepare_fitted_example(
    example: dict[str, Any],
    tokenizer: Any,
    config: dict[str, Any],
    prompt_budget: int,
) -> tuple[dict[str, Any], bool]:
    fitted_example, trimmed = _fit_context_to_prompt_budget(example, tokenizer, config, prompt_budget)
    if _count_text_tokens(tokenizer, _build_prompt_text(tokenizer, config, fitted_example)) <= prompt_budget:
        return fitted_example, trimmed

    qa_only_example = dict(fitted_example)
    qa_only_example["context"] = ""
    qa_only_example["context_source"] = "qa_only"
    qa_only_example["context_url"] = ""
    return qa_only_example, True


def _fetch_keboola_context(filename: str, domain_config: dict[str, Any]) -> dict[str, Any]:
    base_url = domain_config["context_base_url"].rstrip("/")
    timeout_seconds = int(domain_config["context_fetch_timeout_seconds"])
    cache_version = str(domain_config.get("context_cache_version", "v1"))
    for url in _keboola_doc_url_candidates(filename, base_url):
        request = Request(url, headers={"User-Agent": "LoRA-domain-adapt/1.0"})
        try:
            with urlopen(request, timeout=timeout_seconds) as response:
                html_text = response.read().decode("utf-8", errors="replace")
        except (HTTPError, URLError, TimeoutError):
            continue

        title = _extract_document_title(html_text)
        raw_context = _strip_html_to_text(html_text)
        context_text = _clean_context_text(
            _truncate_context(raw_context, domain_config.get("context_max_chars")),
            title,
            domain_config,
        )
        if context_text:
            return {
                "status": "ok",
                "url": url,
                "context": context_text,
                "source": "keboola_docs",
                "cache_version": cache_version,
            }

    return {
        "status": "missing",
        "url": "",
        "context": "",
        "source": "qa_only",
        "cache_version": cache_version,
    }


def _attach_external_context(
    split_records: dict[str, list[dict[str, Any]]],
    config: dict[str, Any],
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    domain_config = config["data"]["domain"]
    if domain_config.get("context_strategy") != "keboola_docs_by_filename":
        return split_records, {
            "strategy": "disabled",
            "resolved_records": 0,
            "total_records": sum(len(records) for records in split_records.values()),
            "resolved_ratio": 0.0,
            "unique_documents": 0,
            "resolved_documents": 0,
        }

    all_records = [record for records in split_records.values() for record in records]
    filenames = sorted(
        {
            record["filename"]
            for record in all_records
            if record.get("filename") and not record.get("context")
        }
    )
    cache_path = _context_cache_path(config)
    cache_version = _context_cache_version(config)
    context_cache = _load_context_cache(cache_path)
    missing_filenames = [
        filename
        for filename in filenames
        if not _is_valid_context_cache_entry(context_cache.get(filename, {}), cache_version)
    ]

    if missing_filenames:
        max_workers = max(1, int(domain_config.get("context_fetch_workers", 8)))
        print(
            f"[context] fetching Keboola docs for {len(missing_filenames)} documents "
            f"(cache miss, workers={max_workers})"
        )
        fetched = 0
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_map = {
                executor.submit(_fetch_keboola_context, filename, domain_config): filename
                for filename in missing_filenames
            }
            for future in concurrent.futures.as_completed(future_map):
                filename = future_map[future]
                try:
                    context_cache[filename] = future.result()
                except Exception as exc:
                    context_cache[filename] = {
                        "status": "error",
                        "url": "",
                        "context": "",
                        "source": "qa_only",
                        "error": str(exc),
                        "cache_version": cache_version,
                    }
                fetched += 1
                if fetched == 1 or fetched % 10 == 0 or fetched == len(missing_filenames):
                    print(f"[context] fetched {fetched}/{len(missing_filenames)}")
        _write_context_cache(cache_path, context_cache)

    cache_changed = False
    for filename in filenames:
        sanitized_entry, changed = _sanitize_cache_entry(
            context_cache.get(filename, {}),
            cache_version,
            domain_config,
        )
        context_cache[filename] = sanitized_entry
        cache_changed = cache_changed or changed
    if cache_changed:
        _write_context_cache(cache_path, context_cache)

    resolved_records = 0
    resolved_documents = 0
    for filename in filenames:
        cached = context_cache.get(filename, {})
        if _is_valid_context_cache_entry(cached, cache_version):
            resolved_documents += 1

    for records in split_records.values():
        for record in records:
            if record.get("context"):
                cleaned_context = _clean_context_text(
                    record["context"],
                    _guess_context_title(record["context"]),
                    domain_config,
                )
                if cleaned_context:
                    record["context"] = cleaned_context
                    record["context_source"] = "dataset"
                    record["context_url"] = ""
                    resolved_records += 1
                else:
                    record["context"] = ""
                    record["context_source"] = "qa_only"
                    record["context_url"] = ""
                continue
            cached = context_cache.get(record.get("filename", ""), {})
            if _is_valid_context_cache_entry(cached, cache_version):
                cleaned_context = _clean_context_text(
                    _coerce_text(cached.get("context")),
                    _guess_context_title(_coerce_text(cached.get("context"))),
                    domain_config,
                )
                if cleaned_context:
                    record["context"] = cleaned_context
                    record["context_source"] = cached.get("source", "keboola_docs")
                    record["context_url"] = cached.get("url", "")
                    resolved_records += 1
                else:
                    record["context"] = ""
                    record["context_source"] = "qa_only"
                    record["context_url"] = ""
            else:
                record["context"] = ""
                record["context_source"] = "qa_only"
                record["context_url"] = ""

    total_records = max(1, len(all_records))
    return split_records, {
        "strategy": domain_config["context_strategy"],
        "resolved_records": resolved_records,
        "total_records": len(all_records),
        "resolved_ratio": resolved_records / total_records,
        "unique_documents": len(filenames),
        "resolved_documents": resolved_documents,
        "cache_path": cache_path,
        "cache_version": cache_version,
    }


def summarize_domain_context(examples: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(examples)
    resolved = sum(1 for example in examples if has_distinct_context(example))
    sources: dict[str, int] = {}
    for example in examples:
        source = _coerce_text(example.get("context_source")) or "unknown"
        sources[source] = sources.get(source, 0) + 1
    return {
        "resolved_records": resolved,
        "total_records": total,
        "resolved_ratio": (resolved / total) if total else 0.0,
        "source_counts": sources,
    }


def _snapshot_metadata(config: dict[str, Any], context_meta: dict[str, Any]) -> dict[str, Any]:
    domain_config = config["data"]["domain"]
    return {
        "dataset_name": resolve_domain_dataset_name(config),
        "dataset_path": domain_config["path"],
        "config_name": domain_config.get("config_name"),
        "split_seed": domain_config["split_seed"],
        "split_ratio": list(domain_config["split_ratio"]),
        "group_by_field": domain_config.get("group_by_field"),
        "context_strategy": domain_config.get("context_strategy"),
        "context_cache_version": _context_cache_version(config),
        "prepared_snapshot_version": _prepared_snapshot_version(config),
        "context_meta": dict(context_meta),
    }


def _is_valid_prepared_snapshot(snapshot: dict[str, Any], config: dict[str, Any]) -> bool:
    if not snapshot:
        return False
    metadata = snapshot.get("metadata", {})
    splits = snapshot.get("splits", {})
    required_splits = {"train", "validation", "test"}
    if not isinstance(metadata, dict) or not required_splits.issubset(set(splits)):
        return False

    domain_config = config["data"]["domain"]
    return (
        metadata.get("dataset_name") == resolve_domain_dataset_name(config)
        and metadata.get("dataset_path") == domain_config["path"]
        and metadata.get("config_name") == domain_config.get("config_name")
        and metadata.get("split_seed") == domain_config["split_seed"]
        and metadata.get("split_ratio") == list(domain_config["split_ratio"])
        and metadata.get("group_by_field") == domain_config.get("group_by_field")
        and metadata.get("context_strategy") == domain_config.get("context_strategy")
        and metadata.get("context_cache_version") == _context_cache_version(config)
        and metadata.get("prepared_snapshot_version") == _prepared_snapshot_version(config)
    )


def _limit_domain_splits(
    split_records: dict[str, list[dict[str, Any]]],
    config: dict[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    domain = config["data"]["domain"]
    return {
        "train": _limit_records(split_records["train"], domain.get("max_train_samples")),
        "validation": _limit_records(split_records["validation"], domain.get("max_validation_samples")),
        "test": _limit_records(split_records["test"], domain.get("max_test_samples")),
    }


def _build_full_domain_splits(config: dict[str, Any]) -> tuple[dict[str, list[dict[str, Any]]], dict[str, str | None], dict[str, Any]]:
    load_dataset, _ = _require_datasets()
    dataset = load_dataset(config["data"]["domain"]["path"], name=config["data"]["domain"].get("config_name"))

    if {"train", "validation", "test"}.issubset(set(dataset.keys())):
        split_map = {
            "train": dataset["train"],
            "validation": dataset["validation"],
            "test": dataset["test"],
        }
    else:
        split_map = _reshuffle_splits(dict(dataset), config)

    sample_record = split_map["train"][0]
    field_map = _infer_domain_field_map(sample_record, config)
    normalized_splits = {
        "train": [
            _normalize_domain_example(example, field_map, index)
            for index, example in enumerate(split_map["train"])
        ],
        "validation": [
            _normalize_domain_example(example, field_map, index)
            for index, example in enumerate(split_map["validation"])
        ],
        "test": [
            _normalize_domain_example(example, field_map, index)
            for index, example in enumerate(split_map["test"])
        ],
    }
    enriched_splits, context_meta = _attach_external_context(normalized_splits, config)
    return enriched_splits, field_map, context_meta


def materialize_domain_snapshot(config: dict[str, Any]) -> dict[str, Any]:
    snapshot_config = copy.deepcopy(config)
    snapshot_config["data"]["domain"]["max_train_samples"] = None
    snapshot_config["data"]["domain"]["max_validation_samples"] = None
    snapshot_config["data"]["domain"]["max_test_samples"] = None

    full_splits, field_map, context_meta = _build_full_domain_splits(snapshot_config)
    snapshot_payload = {
        "metadata": _snapshot_metadata(snapshot_config, context_meta),
        "field_map": dict(field_map),
        "splits": full_splits,
    }
    snapshot_path = _prepared_snapshot_path(snapshot_config)
    _write_json_dict(snapshot_path, snapshot_payload)
    return {
        "snapshot_path": snapshot_path,
        "prepared_snapshot_version": _prepared_snapshot_version(snapshot_config),
        "context_cache_version": _context_cache_version(snapshot_config),
        "split_sizes": {name: len(records) for name, records in full_splits.items()},
        "context_meta": dict(context_meta),
    }


def load_domain_splits(config: dict[str, Any]) -> dict[str, Any]:
    domain = config["data"]["domain"]
    cache_key = _domain_cache_key(domain)
    if cache_key not in _DOMAIN_SPLITS_CACHE:
        use_snapshot = bool(domain.get("prefer_prepared_snapshot", True))
        snapshot_path = _prepared_snapshot_path(config)
        snapshot_payload = _load_json_dict(snapshot_path) if use_snapshot else {}

        if use_snapshot and _is_valid_prepared_snapshot(snapshot_payload, config):
            full_splits = {
                split_name: list(snapshot_payload["splits"][split_name])
                for split_name in ("train", "validation", "test")
            }
            field_map = dict(snapshot_payload.get("field_map", {}))
            context_meta = dict(snapshot_payload.get("metadata", {}).get("context_meta", {}))
            print(f"[data] using prepared domain snapshot: {snapshot_path}")
        else:
            if use_snapshot:
                print(f"[data] building prepared domain snapshot: {snapshot_path}")
                snapshot_info = materialize_domain_snapshot(config)
                snapshot_payload = _load_json_dict(snapshot_info["snapshot_path"])
                print(
                    "[data] prepared snapshot saved "
                    f"(train={snapshot_info['split_sizes']['train']}, "
                    f"validation={snapshot_info['split_sizes']['validation']}, "
                    f"test={snapshot_info['split_sizes']['test']})"
                )
                full_splits = {
                    split_name: list(snapshot_payload["splits"][split_name])
                    for split_name in ("train", "validation", "test")
                }
                field_map = dict(snapshot_payload.get("field_map", {}))
                context_meta = dict(snapshot_payload.get("metadata", {}).get("context_meta", {}))
            else:
                full_splits, field_map, context_meta = _build_full_domain_splits(config)

        limited_splits = _limit_domain_splits(full_splits, config)
        _DOMAIN_SPLITS_CACHE[cache_key] = {
            "train": limited_splits["train"],
            "validation": limited_splits["validation"],
            "test": limited_splits["test"],
            "field_map": dict(field_map),
            "context_meta": context_meta,
        }
        if context_meta.get("strategy") != "disabled":
            print(
                "[context] coverage "
                f"{context_meta['resolved_records']}/{context_meta['total_records']} "
                f"({context_meta['resolved_ratio']:.1%})"
            )

    cached = _DOMAIN_SPLITS_CACHE[cache_key]
    return {
        "train": list(cached["train"]),
        "validation": list(cached["validation"]),
        "test": list(cached["test"]),
        "field_map": dict(cached["field_map"]),
        "context_meta": dict(cached.get("context_meta", {})),
    }


def has_distinct_context(example: dict[str, Any]) -> bool:
    context = _coerce_text(example.get("context"))
    question = _coerce_text(example.get("question"))
    return bool(context) and context != question


def infer_domain_input_mode(examples: list[dict[str, Any]]) -> str:
    if any(has_distinct_context(example) for example in examples):
        return "context_qa"
    return "qa_only"


def format_domain_user_prompt(config: dict[str, Any], example: dict[str, Any]) -> str:
    if has_distinct_context(example):
        return config["prompt"]["domain_user_template"].format(
            context=example["context"],
            question=example["question"],
            filename=example.get("filename", ""),
        )
    return config["prompt"]["domain_question_only_template"].format(
        question=example["question"],
        filename=example.get("filename", ""),
    )


def build_chat_messages(
    system_prompt: str,
    user_prompt: str,
    answer: str | None = None,
) -> list[dict[str, str]]:
    messages = [{"role": "system", "content": system_prompt}]
    messages.append({"role": "user", "content": user_prompt})
    if answer is not None:
        messages.append({"role": "assistant", "content": answer})
    return messages


def apply_chat_template(tokenizer: Any, messages: list[dict[str, str]], add_generation_prompt: bool) -> str:
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )

    rendered = []
    for message in messages:
        rendered.append(f"{message['role'].upper()}:\n{message['content']}\n")
    if add_generation_prompt:
        rendered.append("ASSISTANT:\n")
    return "\n".join(rendered)


def prepare_supervised_records(
    examples: list[dict[str, Any]],
    tokenizer: Any,
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    prepared: list[dict[str, Any]] = []
    max_seq_len = int(config["model"]["max_seq_len"])
    answer_token_reserve = int(config["data"]["domain"]["answer_token_reserve"])
    eos_text = tokenizer.eos_token or ""

    trimmed_count = 0
    fallback_to_qa_only = 0
    skipped_count = 0
    for example in examples:
        answer_ids = tokenizer(
            f"{example['answer']}{eos_text}",
            add_special_tokens=False,
            truncation=True,
            max_length=answer_token_reserve,
        )["input_ids"]
        answer_budget = max(1, len(answer_ids))
        prompt_budget = max(32, max_seq_len - answer_budget)

        fitted_example, trimmed = _prepare_fitted_example(example, tokenizer, config, prompt_budget)
        if trimmed:
            trimmed_count += 1
        if example.get("context") and not fitted_example.get("context"):
            fallback_to_qa_only += 1

        prompt_text = _build_prompt_text(tokenizer, config, fitted_example)
        prompt_ids = tokenizer(
            prompt_text,
            add_special_tokens=False,
            truncation=False,
        )["input_ids"]
        if len(prompt_ids) >= max_seq_len:
            skipped_count += 1
            continue

        full_text = f"{prompt_text}{example['answer']}{eos_text}"
        encoded = tokenizer(
            full_text,
            add_special_tokens=False,
            truncation=True,
            max_length=max_seq_len,
        )
        input_ids = encoded["input_ids"]
        if len(prompt_ids) >= len(input_ids):
            skipped_count += 1
            continue

        labels = [-100] * len(prompt_ids) + input_ids[len(prompt_ids):]
        prepared.append(
            {
                "id": fitted_example["id"],
                "input_ids": input_ids,
                "attention_mask": encoded["attention_mask"],
                "labels": labels,
            }
        )

    print(
        f"[data] supervised records kept {len(prepared)}/{len(examples)} "
        f"(trimmed={trimmed_count}, qa_only_fallback={fallback_to_qa_only}, skipped={skipped_count})"
    )
    return prepared


def prepare_domain_generation_records(
    examples: list[dict[str, Any]],
    tokenizer: Any,
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    prepared: list[dict[str, Any]] = []
    max_seq_len = int(config["model"]["max_seq_len"])

    trimmed_count = 0
    qa_only_fallback = 0
    for example in examples:
        fitted_example, trimmed = _prepare_fitted_example(example, tokenizer, config, max_seq_len)
        if trimmed:
            trimmed_count += 1
        if example.get("context") and not fitted_example.get("context"):
            qa_only_fallback += 1

        prompt_text = _build_prompt_text(tokenizer, config, fitted_example)
        prepared.append(
            {
                "id": fitted_example["id"],
                "prompt_text": prompt_text,
                "reference_answer": fitted_example["answer"],
                "question": fitted_example["question"],
                "context": fitted_example["context"],
                "filename": fitted_example.get("filename", ""),
                "context_source": fitted_example.get("context_source", ""),
                "context_url": fitted_example.get("context_url", ""),
            }
        )

    print(
        f"[data] generation prompts prepared {len(prepared)} "
        f"(trimmed={trimmed_count}, qa_only_fallback={qa_only_fallback})"
    )
    return prepared


def load_boolq_examples(config: dict[str, Any]) -> list[dict[str, Any]]:
    boolq_config = config["data"]["general"]["boolq"]
    cache_key = _general_cache_key(boolq_config, "boolq")
    if cache_key not in _GENERAL_DATA_CACHE:
        load_dataset, _ = _require_datasets()
        dataset = load_dataset(
            boolq_config["path"],
            name=boolq_config.get("config_name"),
            split=boolq_config["split"],
        )
        examples: list[dict[str, Any]] = []
        for row in dataset:
            examples.append(
                {
                    "question": _coerce_text(row["question"]),
                    "passage": _coerce_text(row["passage"]),
                    "label": "yes" if bool(row["answer"]) else "no",
                }
            )
        _GENERAL_DATA_CACHE[cache_key] = examples
    return _limit_records(_GENERAL_DATA_CACHE[cache_key], boolq_config.get("max_samples"))


def load_piqa_examples(config: dict[str, Any]) -> list[dict[str, Any]]:
    piqa_config = config["data"]["general"]["piqa"]
    cache_key = _general_cache_key(piqa_config, "piqa")
    if cache_key not in _GENERAL_DATA_CACHE:
        load_dataset, _ = _require_datasets()
        try:
            dataset = load_dataset(
                piqa_config["path"],
                name=piqa_config.get("config_name"),
                split=piqa_config["split"],
            )
            examples: list[dict[str, Any]] = []
            for row in dataset:
                label = row.get("label")
                examples.append(
                    {
                        "goal": _coerce_text(row["goal"]),
                        "sol1": _coerce_text(row["sol1"]),
                        "sol2": _coerce_text(row["sol2"]),
                        "label": "A" if int(label) == 0 else "B",
                    }
                )
        except Exception as exc:
            if "Dataset scripts are no longer supported" not in str(exc):
                raise
            print(
                "[data] PIQA loader fallback: datasets remote script is unsupported, "
                "loading from the official PIQA source files instead."
            )
            examples = _load_piqa_examples_manual(piqa_config)
        _GENERAL_DATA_CACHE[cache_key] = examples
    return _limit_records(_GENERAL_DATA_CACHE[cache_key], piqa_config.get("max_samples"))


def _load_text_from_url(url: str) -> str:
    with urlopen(url, timeout=60) as response:
        return response.read().decode("utf-8")


def _load_bytes_from_url(url: str) -> bytes:
    with urlopen(url, timeout=120) as response:
        return response.read()


def _parse_piqa_rows(inputs_text: str, labels_text: str | None) -> list[dict[str, Any]]:
    input_rows = inputs_text.splitlines()
    label_rows = labels_text.splitlines() if labels_text is not None else ["-1"] * len(input_rows)
    examples: list[dict[str, Any]] = []
    for row_text, label_text in zip(input_rows, label_rows):
        row = json.loads(row_text)
        label_value = int(label_text)
        examples.append(
            {
                "goal": _coerce_text(row["goal"]),
                "sol1": _coerce_text(row["sol1"]),
                "sol2": _coerce_text(row["sol2"]),
                "label": "A" if label_value == 0 else "B",
            }
        )
    return examples


def _load_piqa_examples_manual(piqa_config: dict[str, Any]) -> list[dict[str, Any]]:
    split_name = piqa_config["split"]
    train_dev_zip_url = "https://storage.googleapis.com/ai2-mosaic/public/physicaliqa/physicaliqa-train-dev.zip"
    test_url = "https://yonatanbisk.com/piqa/data/tests.jsonl"

    if split_name in {"train", "validation"}:
        archive_bytes = _load_bytes_from_url(train_dev_zip_url)
        with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
            inner_root = "physicaliqa-train-dev"
            if split_name == "train":
                input_name = f"{inner_root}/train.jsonl"
                label_name = f"{inner_root}/train-labels.lst"
            else:
                input_name = f"{inner_root}/dev.jsonl"
                label_name = f"{inner_root}/dev-labels.lst"

            inputs_text = archive.read(input_name).decode("utf-8")
            labels_text = archive.read(label_name).decode("utf-8")
        examples = _parse_piqa_rows(inputs_text, labels_text)
    elif split_name == "test":
        inputs_text = _load_text_from_url(test_url)
        examples = _parse_piqa_rows(inputs_text, labels_text=None)
    else:
        raise ValueError(f"Unsupported PIQA split: {split_name}")

    return examples


def load_wikitext_texts(config: dict[str, Any]) -> list[str]:
    wikitext_config = config["data"]["general"]["wikitext"]
    cache_key = _general_cache_key(wikitext_config, "wikitext")
    if cache_key not in _GENERAL_DATA_CACHE:
        load_dataset, _ = _require_datasets()
        dataset = load_dataset(
            wikitext_config["path"],
            name=wikitext_config.get("config_name"),
            split=wikitext_config["split"],
        )
        texts = [_coerce_text(row["text"]) for row in dataset if _coerce_text(row["text"])]
        _GENERAL_DATA_CACHE[cache_key] = texts
    cached_texts = list(_GENERAL_DATA_CACHE[cache_key])
    limit = wikitext_config.get("max_samples")
    if limit is None:
        return cached_texts
    return cached_texts[: min(limit, len(cached_texts))]
