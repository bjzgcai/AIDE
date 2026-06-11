import contextlib
import contextvars
import json
import os
import re
import time
import uuid
from collections.abc import Iterator
from datetime import UTC, datetime
from typing import Any

from loguru import logger

_ACTIVE_METRICS_TRACKER = contextvars.ContextVar("active_metrics_tracker", default=None)
_ACTIVE_METRICS_SCOPES = contextvars.ContextVar("active_metrics_scopes", default=())


def _utcnow_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _load_json(path: str) -> Any:
    if not path or not os.path.exists(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        logger.warning(f"Failed to load metrics JSON from {path}: {exc}")
        return None


def _save_json(path: str, data: Any) -> None:
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)
    tmp_path = f"{path}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass


def _json_safe(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return str(value)


def _safe_int(value: Any) -> int:
    try:
        return int(value or 0)
    except Exception:
        return 0


def _safe_float(value: Any) -> float:
    try:
        return float(value or 0.0)
    except Exception:
        return 0.0


def _extract_usage_fields(
    usage: Any,
    *,
    prompt_tokens: int,
    completion_tokens: int,
    total_tokens: int,
) -> dict[str, Any]:
    if usage is None:
        return {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "estimated": True,
        }

    if hasattr(usage, "model_dump"):
        usage_dict = usage.model_dump()
    elif isinstance(usage, dict):
        usage_dict = usage
    else:
        usage_dict = {
            "prompt_tokens": getattr(usage, "prompt_tokens", 0),
            "completion_tokens": getattr(usage, "completion_tokens", 0),
            "total_tokens": getattr(usage, "total_tokens", 0),
        }

    normalized_prompt_tokens = _safe_int(usage_dict.get("prompt_tokens", prompt_tokens))
    normalized_completion_tokens = _safe_int(
        usage_dict.get("completion_tokens", completion_tokens)
    )
    normalized_total_tokens = _safe_int(usage_dict.get("total_tokens", total_tokens))
    if normalized_total_tokens == 0:
        normalized_total_tokens = (
            normalized_prompt_tokens + normalized_completion_tokens
        )

    return {
        "prompt_tokens": normalized_prompt_tokens,
        "completion_tokens": normalized_completion_tokens,
        "total_tokens": normalized_total_tokens,
        "estimated": False,
    }


def get_active_metrics_tracker():
    return _ACTIVE_METRICS_TRACKER.get()


def get_active_metrics_scopes() -> tuple[str, ...]:
    return tuple(_ACTIVE_METRICS_SCOPES.get())


@contextlib.contextmanager
def activate_metrics_tracker(tracker) -> Iterator[Any]:
    token = _ACTIVE_METRICS_TRACKER.set(tracker)
    try:
        yield tracker
    finally:
        _ACTIVE_METRICS_TRACKER.reset(token)


_DEFAULT_MIN_PERSIST_INTERVAL_SEC = float(
    os.environ.get("STAGE_METRICS_MIN_PERSIST_INTERVAL_SEC", "2.0") or 2.0
)


class StageCostTracker:
    def __init__(
        self,
        path: str,
        pricing_path: str = "",
        min_persist_interval_sec: float | None = None,
    ):
        self.path = path
        self.summary_path = os.path.join(
            os.path.dirname(path), "stage_metrics_summary.txt"
        )
        self.pricing_path = pricing_path
        self.pricing = self._load_pricing(pricing_path)

        loaded = _load_json(path)
        if not isinstance(loaded, dict):
            loaded = {}

        stages = loaded.get("stages")
        if not isinstance(stages, dict):
            stages = {}

        self.data = {
            "created_at": loaded.get("created_at", _utcnow_iso()),
            "updated_at": loaded.get("updated_at", _utcnow_iso()),
            "pricing_path": pricing_path or loaded.get("pricing_path", ""),
            "stages": stages,
        }

        # Throttle disk writes so a hot path that updates the tracker on
        # every iteration (e.g. `record_named_duration` from the per-row
        # processing loop) does not rewrite the entire stage_metrics.json
        # to a slow shared filesystem on every event. Real persist happens
        # at most once every `min_persist_interval_sec` seconds; pending
        # changes are still flushed by `persist(force=True)`, which is
        # called automatically when a metrics scope exits.
        self.min_persist_interval_sec = (
            min_persist_interval_sec
            if min_persist_interval_sec is not None
            else _DEFAULT_MIN_PERSIST_INTERVAL_SEC
        )
        self._last_persist_at = 0.0
        self._dirty = False

    def _load_pricing(self, pricing_path: str) -> dict[str, Any]:
        if not pricing_path:
            return {}
        loaded = _load_json(pricing_path)
        return loaded if isinstance(loaded, dict) else {}

    def _empty_stage(self, name: str) -> dict[str, Any]:
        return {
            "name": name,
            "wall_time_sec": 0.0,
            "api_time_sec": 0.0,
            "api_calls": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "estimated_cost_usd": 0.0,
            "cache_hits": 0,
            "cache_misses": 0,
            "runs": 0,
            "counters": {},
            "metadata": {},
            "llm_models": {},
            "last_updated_at": _utcnow_iso(),
        }

    def _ensure_stage(self, name: str) -> dict[str, Any]:
        stages = self.data["stages"]
        if name not in stages or not isinstance(stages[name], dict):
            stages[name] = self._empty_stage(name)
        stage = stages[name]
        stage.setdefault("name", name)
        stage.setdefault("wall_time_sec", 0.0)
        stage.setdefault("api_time_sec", 0.0)
        stage.setdefault("api_calls", 0)
        stage.setdefault("prompt_tokens", 0)
        stage.setdefault("completion_tokens", 0)
        stage.setdefault("total_tokens", 0)
        stage.setdefault("estimated_cost_usd", 0.0)
        stage.setdefault("cache_hits", 0)
        stage.setdefault("cache_misses", 0)
        stage.setdefault("runs", 0)
        stage.setdefault("counters", {})
        stage.setdefault("metadata", {})
        stage.setdefault("llm_models", {})
        stage.setdefault("last_updated_at", _utcnow_iso())
        return stage

    def _touch_stage(self, stage: dict[str, Any]) -> None:
        stage["last_updated_at"] = _utcnow_iso()

    def _estimate_cost(
        self, model: str, prompt_tokens: int, completion_tokens: int
    ) -> float:
        if not self.pricing:
            return 0.0
        pricing = self.pricing.get(model) or self.pricing.get(model.lower())
        if not isinstance(pricing, dict):
            return 0.0

        input_price = _safe_float(
            pricing.get(
                "input_per_million_tokens_usd",
                pricing.get("prompt_per_million_tokens_usd", 0.0),
            )
        )
        output_price = _safe_float(
            pricing.get(
                "output_per_million_tokens_usd",
                pricing.get("completion_per_million_tokens_usd", 0.0),
            )
        )
        return (prompt_tokens / 1_000_000.0) * input_price + (
            completion_tokens / 1_000_000.0
        ) * output_price

    @contextlib.contextmanager
    def scope(
        self, name: str, metadata: dict[str, Any] | None = None
    ) -> Iterator[dict[str, Any]]:
        stage = self._ensure_stage(name)
        stage["runs"] += 1
        if metadata:
            stage["metadata"].update(_json_safe(metadata))
        self._touch_stage(stage)

        current_scopes = list(get_active_metrics_scopes())
        current_scopes.append(name)
        token = _ACTIVE_METRICS_SCOPES.set(tuple(current_scopes))
        start = time.perf_counter()
        try:
            yield stage
        finally:
            elapsed = time.perf_counter() - start
            stage = self._ensure_stage(name)
            stage["wall_time_sec"] += elapsed
            self._touch_stage(stage)
            _ACTIVE_METRICS_SCOPES.reset(token)
            self.persist(force=True)

    def record_cache_event(
        self, cache_hit: bool, scope_names: tuple[str, ...] | None = None
    ) -> None:
        target_scopes = scope_names or get_active_metrics_scopes()
        for scope_name in dict.fromkeys(target_scopes):
            stage = self._ensure_stage(scope_name)
            if cache_hit:
                stage["cache_hits"] += 1
            else:
                stage["cache_misses"] += 1
            self._touch_stage(stage)
        self.persist()

    def record_counter(
        self,
        counter_name: str,
        delta: int | float = 1,
        scope_names: tuple[str, ...] | None = None,
    ) -> None:
        target_scopes = scope_names or get_active_metrics_scopes()
        for scope_name in dict.fromkeys(target_scopes):
            stage = self._ensure_stage(scope_name)
            counters = stage["counters"]
            counters[counter_name] = counters.get(counter_name, 0) + delta
            self._touch_stage(stage)
        self.persist()

    def record_named_duration(
        self,
        stage_name: str,
        duration_sec: float,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        if duration_sec <= 0:
            return
        stage = self._ensure_stage(stage_name)
        stage["wall_time_sec"] += duration_sec
        if metadata:
            stage["metadata"].update(_json_safe(metadata))
        self._touch_stage(stage)
        self.persist()

    def record_metadata(
        self, metadata: dict[str, Any], scope_names: tuple[str, ...] | None = None
    ) -> None:
        target_scopes = scope_names or get_active_metrics_scopes()
        for scope_name in dict.fromkeys(target_scopes):
            stage = self._ensure_stage(scope_name)
            stage["metadata"].update(_json_safe(metadata))
            self._touch_stage(stage)
        self.persist()

    def record_llm_usage(
        self,
        *,
        model: str,
        api_time_sec: float,
        usage: Any,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        total_tokens: int = 0,
        scope_names: tuple[str, ...] | None = None,
    ) -> None:
        normalized = _extract_usage_fields(
            usage,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
        )
        target_scopes = scope_names or get_active_metrics_scopes()
        if not target_scopes:
            target_scopes = ("unscoped_llm",)

        estimated_cost = self._estimate_cost(
            model,
            normalized["prompt_tokens"],
            normalized["completion_tokens"],
        )

        for scope_name in dict.fromkeys(target_scopes):
            stage = self._ensure_stage(scope_name)
            stage["api_time_sec"] += api_time_sec
            stage["api_calls"] += 1
            stage["prompt_tokens"] += normalized["prompt_tokens"]
            stage["completion_tokens"] += normalized["completion_tokens"]
            stage["total_tokens"] += normalized["total_tokens"]
            stage["estimated_cost_usd"] += estimated_cost
            if normalized["estimated"]:
                counters = stage["counters"]
                counters["estimated_usage_calls"] = (
                    counters.get("estimated_usage_calls", 0) + 1
                )

            model_stats = stage["llm_models"].setdefault(
                model,
                {
                    "api_calls": 0,
                    "api_time_sec": 0.0,
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                    "estimated_cost_usd": 0.0,
                },
            )
            model_stats["api_calls"] += 1
            model_stats["api_time_sec"] += api_time_sec
            model_stats["prompt_tokens"] += normalized["prompt_tokens"]
            model_stats["completion_tokens"] += normalized["completion_tokens"]
            model_stats["total_tokens"] += normalized["total_tokens"]
            model_stats["estimated_cost_usd"] += estimated_cost
            self._touch_stage(stage)

        self.persist()

    def record_llm_failure(
        self,
        *,
        model: str,
        api_time_sec: float,
        scope_names: tuple[str, ...] | None = None,
    ) -> None:
        target_scopes = scope_names or get_active_metrics_scopes()
        if not target_scopes:
            target_scopes = ("unscoped_llm",)

        for scope_name in dict.fromkeys(target_scopes):
            stage = self._ensure_stage(scope_name)
            stage["api_time_sec"] += api_time_sec
            counters = stage["counters"]
            counters["failed_api_calls"] = counters.get("failed_api_calls", 0) + 1

            model_stats = stage["llm_models"].setdefault(
                model,
                {
                    "api_calls": 0,
                    "api_time_sec": 0.0,
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                    "estimated_cost_usd": 0.0,
                },
            )
            model_stats["api_time_sec"] += api_time_sec
            model_stats["failed_api_calls"] = model_stats.get("failed_api_calls", 0) + 1
            self._touch_stage(stage)

        self.persist()

    def persist(self, force: bool = False) -> None:
        now = time.perf_counter()
        if (
            not force
            and self.min_persist_interval_sec > 0
            and (now - self._last_persist_at) < self.min_persist_interval_sec
        ):
            self._dirty = True
            return
        self.data["updated_at"] = _utcnow_iso()
        _save_json(self.path, self.data)
        self._write_summary()
        self._last_persist_at = now
        self._dirty = False

    def _write_summary(self) -> None:
        lines = [
            f"created_at={self.data.get('created_at', '')}",
            f"updated_at={self.data.get('updated_at', '')}",
        ]
        if self.data.get("pricing_path"):
            lines.append(f"pricing_path={self.data['pricing_path']}")
        lines.append("")

        for scope_name in sorted(self.data["stages"]):
            stage = self.data["stages"][scope_name]
            counters = stage.get("counters", {})
            lines.append(f"[{scope_name}]")
            lines.append(f"wall_time_sec={stage.get('wall_time_sec', 0.0):.6f}")
            lines.append(f"api_time_sec={stage.get('api_time_sec', 0.0):.6f}")
            lines.append(f"api_calls={stage.get('api_calls', 0)}")
            lines.append(f"prompt_tokens={stage.get('prompt_tokens', 0)}")
            lines.append(f"completion_tokens={stage.get('completion_tokens', 0)}")
            lines.append(f"total_tokens={stage.get('total_tokens', 0)}")
            lines.append(
                f"estimated_cost_usd={stage.get('estimated_cost_usd', 0.0):.8f}"
            )
            lines.append(f"cache_hits={stage.get('cache_hits', 0)}")
            lines.append(f"cache_misses={stage.get('cache_misses', 0)}")
            if counters:
                lines.append(
                    "counters="
                    + json.dumps(counters, ensure_ascii=False, sort_keys=True)
                )
            lines.append("")

        with open(self.summary_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines).rstrip() + "\n")


@contextlib.contextmanager
def metrics_scope(name: str, metadata: dict[str, Any] | None = None) -> Iterator[None]:
    tracker = get_active_metrics_tracker()
    if tracker is None:
        yield
        return
    with tracker.scope(name, metadata=metadata):
        yield


def record_cache_event(cache_hit: bool) -> None:
    tracker = get_active_metrics_tracker()
    if tracker is None:
        return
    tracker.record_cache_event(cache_hit)


def record_counter(counter_name: str, delta: int | float = 1) -> None:
    tracker = get_active_metrics_tracker()
    if tracker is None:
        return
    tracker.record_counter(counter_name, delta)


def record_named_duration(
    stage_name: str, duration_sec: float, metadata: dict[str, Any] | None = None
) -> None:
    tracker = get_active_metrics_tracker()
    if tracker is None:
        return
    tracker.record_named_duration(stage_name, duration_sec, metadata=metadata)


def record_metadata(metadata: dict[str, Any]) -> None:
    tracker = get_active_metrics_tracker()
    if tracker is None:
        return
    tracker.record_metadata(metadata)


def make_dataset_key(dataset_info: Any) -> str:
    subset = getattr(dataset_info, "subset", "") or "default"
    return f"{getattr(dataset_info, 'name', 'unknown')}::{subset}"


def make_dataset_scope_slug(dataset_info: Any) -> str:
    base = make_dataset_key(dataset_info)
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", base).strip("._")
    return slug[:180] or "dataset"


class DatasetRunStateStore:
    def __init__(self, path: str):
        self.path = path
        loaded = _load_json(path)
        if not isinstance(loaded, dict):
            loaded = {}

        datasets = loaded.get("datasets")
        if not isinstance(datasets, dict):
            datasets = {}

        self.data = {
            "created_at": loaded.get("created_at", _utcnow_iso()),
            "updated_at": loaded.get("updated_at", _utcnow_iso()),
            "datasets": datasets,
        }

    def _ensure_entry(self, dataset_info: Any) -> dict[str, Any]:
        dataset_key = make_dataset_key(dataset_info)
        dataset_name = getattr(dataset_info, "name", "unknown")
        subset = getattr(dataset_info, "subset", "") or "default"
        scope_name = f"organization.dataset::{make_dataset_scope_slug(dataset_info)}"

        entry = self.data["datasets"].setdefault(
            dataset_key,
            {
                "dataset_key": dataset_key,
                "dataset_name": dataset_name,
                "subset": subset,
                "scope_name": scope_name,
                "status": "pending",
                "attempts": 0,
                "best_effort_used": False,
                "chunk_count": 0,
                "output_path": "",
                "code_path": "",
                "last_error": "",
                "last_updated_at": _utcnow_iso(),
            },
        )
        entry.setdefault("dataset_key", dataset_key)
        entry.setdefault("dataset_name", dataset_name)
        entry.setdefault("subset", subset)
        entry.setdefault("scope_name", scope_name)
        entry.setdefault("status", "pending")
        entry.setdefault("attempts", 0)
        entry.setdefault("best_effort_used", False)
        entry.setdefault("chunk_count", 0)
        entry.setdefault("output_path", "")
        entry.setdefault("code_path", "")
        entry.setdefault("last_error", "")
        entry.setdefault("last_updated_at", _utcnow_iso())
        return entry

    def get(self, dataset_info_or_key: Any) -> dict[str, Any] | None:
        if isinstance(dataset_info_or_key, str):
            dataset_key = dataset_info_or_key
        else:
            dataset_key = make_dataset_key(dataset_info_or_key)
        entry = self.data["datasets"].get(dataset_key)
        return entry if isinstance(entry, dict) else None

    def update(self, dataset_info: Any, **fields: Any) -> dict[str, Any]:
        entry = self._ensure_entry(dataset_info)
        for key, value in fields.items():
            entry[key] = _json_safe(value)
        entry["last_updated_at"] = _utcnow_iso()
        self.data["updated_at"] = _utcnow_iso()
        self.persist()
        return entry

    def increment(
        self, dataset_info: Any, field: str, delta: int = 1
    ) -> dict[str, Any]:
        entry = self._ensure_entry(dataset_info)
        entry[field] = _safe_int(entry.get(field, 0)) + delta
        entry["last_updated_at"] = _utcnow_iso()
        self.data["updated_at"] = _utcnow_iso()
        self.persist()
        return entry

    def persist(self) -> None:
        _save_json(self.path, self.data)
