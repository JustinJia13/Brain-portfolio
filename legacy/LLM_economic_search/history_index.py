"""Read-only history scanner and novelty index builder."""

from __future__ import annotations

import csv
import json
import re
import sqlite3
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .schemas import normalize_expression, semantic_normalize_expression, stable_hash


ROOT = Path(__file__).resolve().parent.parent
PACKAGE_DIR = Path(__file__).resolve().parent
DEFAULT_ARTIFACTS_DIR = PACKAGE_DIR / "artifacts"
DEFAULT_INDEX_PATH = DEFAULT_ARTIFACTS_DIR / "history_index.sqlite"
DEFAULT_SUMMARY_PATH = DEFAULT_ARTIFACTS_DIR / "history_summary.json"
DEFAULT_FIELD_CSV = ROOT / "all_fields.csv"
DEFAULT_LOG_PATHS = (
    ROOT / "artifacts" / "sim" / "master.jsonl",
    ROOT / "artifacts" / "workshop" / "workshop_sim.jsonl",
    ROOT / "alpha_llm" / "artifacts" / "llm_sim.jsonl",
    ROOT / "artifacts" / "portfolio_admission" / "ranked.jsonl",
)

_OP_RE = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\(")


def _json_dumps(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _json_loads(value: str | None, default: Any) -> Any:
    if not value:
        return default
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return default


def _read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def _regime_key(record: dict[str, Any]) -> str:
    settings = record.get("settings") or record.get("settings_override") or {}
    from wq_alpha.brain.submit import _regime_key as _canonical_regime_key

    return _canonical_regime_key(settings)


def _record_delay(record: dict[str, Any]) -> int:
    for source in (
        {"delay": record.get("target_delay")},
        record.get("settings") or {},
        record.get("settings_override") or {},
        {"delay": record.get("admission_delay")},
    ):
        try:
            return int(source.get("delay", 1))
        except (AttributeError, TypeError, ValueError):
            continue
    return 1


def _failed_checks(record: dict[str, Any]) -> list[str]:
    checks = record.get("checks") or (record.get("raw_alpha_detail") or {}).get("is", {}).get("checks") or []
    return sorted(
        str(c.get("name"))
        for c in checks
        if isinstance(c, dict) and c.get("result") == "FAIL" and c.get("name")
    )


def _is_check_passing(record: dict[str, Any]) -> bool:
    checks = record.get("checks") or []
    return bool(checks) and not any(c.get("result") == "FAIL" for c in checks)


def _operator_skeleton(record: dict[str, Any], expression: str) -> list[str]:
    template = str(record.get("template_id") or "")
    if template:
        parts = [p for p in template.split("|") if p]
        if parts:
            return parts
    return _OP_RE.findall(expression)


def _category_key(value: str) -> str:
    v = (value or "").strip().lower()
    aliases = {
        "price volume": "price_volume",
        "social media": "social_media",
        "options": "option",
    }
    return aliases.get(v, v)


@dataclass(frozen=True)
class FieldMeta:
    field_id: str
    dataset_id: str
    category: str
    field_name: str
    coverage: float
    alpha_count: int
    cluster_id: str


class FieldCatalog:
    """Small field metadata helper backed by all_fields.csv."""

    def __init__(self, csv_path: Path = DEFAULT_FIELD_CSV) -> None:
        self.csv_path = Path(csv_path)
        self.by_field: dict[str, FieldMeta] = {}
        self.field_ids: set[str] = set()
        self.load()

    def load(self) -> None:
        if not self.csv_path.exists():
            return
        with self.csv_path.open(newline="") as f:
            for row in csv.DictReader(f):
                field_id = row.get("field_id", "")
                if not field_id:
                    continue
                dataset_id = row.get("dataset_id", "")
                category = _category_key(row.get("category", ""))
                field_name = row.get("field_name", "")
                coverage = _safe_float(row.get("coverage"))
                alpha_count = int(_safe_float(row.get("alpha_count")))
                cluster_id = f"{dataset_id}:{_slug(field_name or field_id)}"
                meta = FieldMeta(
                    field_id=field_id,
                    dataset_id=dataset_id,
                    category=category,
                    field_name=field_name,
                    coverage=coverage,
                    alpha_count=alpha_count,
                    cluster_id=cluster_id,
                )
                self.by_field[field_id] = meta
                self.field_ids.add(field_id)

    def extract_fields(self, expression: str) -> list[str]:
        tokens = re.findall(r"\b[A-Za-z_][A-Za-z0-9_]*\b", expression or "")
        seen: set[str] = set()
        fields: list[str] = []
        for token in tokens:
            if token in self.field_ids and token not in seen:
                fields.append(token)
                seen.add(token)
        return fields

    def meta(self, field_id: str) -> FieldMeta | None:
        return self.by_field.get(field_id)


def _safe_float(value: Any) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _slug(value: str) -> str:
    clean = re.sub(r"[^a-zA-Z0-9]+", "_", value.lower()).strip("_")
    clean = re.sub(r"_+", "_", clean)
    return clean[:72] or "unknown"


class HistoryIndexer:
    """Builds the SQLite index and compact JSON summaries."""

    def __init__(
        self,
        *,
        index_path: Path = DEFAULT_INDEX_PATH,
        summary_path: Path = DEFAULT_SUMMARY_PATH,
        field_csv: Path = DEFAULT_FIELD_CSV,
        log_paths: Iterable[Path] = DEFAULT_LOG_PATHS,
    ) -> None:
        self.index_path = Path(index_path)
        self.summary_path = Path(summary_path)
        self.field_catalog = FieldCatalog(Path(field_csv))
        self.log_paths = [Path(p) for p in log_paths]

    def build(self) -> dict[str, Any]:
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        if self.index_path.exists():
            self.index_path.unlink()
        conn = sqlite3.connect(self.index_path)
        try:
            self._create_schema(conn)
            records = self._scan_records()
            self._insert_records(conn, records)
            summary = self._build_summary(records)
            self.summary_path.parent.mkdir(parents=True, exist_ok=True)
            self.summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
            return summary
        finally:
            conn.close()

    def _create_schema(self, conn: sqlite3.Connection) -> None:
        conn.executescript(
            """
            CREATE TABLE records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_path TEXT NOT NULL,
                expression_hash TEXT NOT NULL,
                expression TEXT NOT NULL,
                normalized_expression TEXT NOT NULL,
                regime TEXT NOT NULL,
                delay INTEGER NOT NULL,
                alpha_id TEXT,
                status TEXT,
                logged_at TEXT,
                dataset_id TEXT,
                category TEXT,
                search_lane TEXT,
                template_id TEXT,
                field_key TEXT,
                skeleton_key TEXT,
                fields_json TEXT NOT NULL,
                clusters_json TEXT NOT NULL,
                skeleton_json TEXT NOT NULL,
                failed_checks_json TEXT NOT NULL,
                grade TEXT,
                is_sharpe REAL,
                is_fitness REAL,
                is_turnover REAL,
                max_self_correlation REAL,
                admission_decision TEXT,
                admission_reason TEXT,
                admission_global_max_corr REAL,
                admission_brain_score_delta REAL,
                admission_delta_pool_sharpe REAL,
                admission_signal TEXT,
                admission_fallback_delta_used INTEGER,
                standalone_pass INTEGER NOT NULL
            );
            CREATE INDEX idx_records_expr_regime ON records(expression_hash, regime);
            CREATE INDEX idx_records_expr_delay ON records(expression_hash, delay);
            CREATE INDEX idx_records_field_key ON records(field_key);
            CREATE INDEX idx_records_skeleton_key ON records(skeleton_key);
            CREATE INDEX idx_records_delay ON records(delay);
            CREATE INDEX idx_records_dataset ON records(dataset_id);
            CREATE INDEX idx_records_admission ON records(admission_decision);
            """
        )

    def _scan_records(self) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        seen: set[tuple[str, str]] = set()
        for path in self.log_paths:
            for raw in _read_jsonl(path):
                expression = normalize_expression(str(raw.get("expression") or ""))
                if not expression:
                    continue
                regime = str(raw.get("regime") or _regime_key(raw))
                delay = _record_delay(raw)
                expression_hash = stable_hash(expression, 20)
                dedup_key = (expression_hash, regime)
                if dedup_key in seen:
                    continue
                seen.add(dedup_key)

                fields = [str(f) for f in (raw.get("fields") or []) if f]
                if not fields:
                    fields = self.field_catalog.extract_fields(expression)
                clusters = [str(c) for c in (raw.get("cluster_ids") or []) if c]
                if not clusters:
                    clusters = [
                        self.field_catalog.meta(f).cluster_id
                        for f in fields
                        if self.field_catalog.meta(f) is not None
                    ]
                skeleton = _operator_skeleton(raw, expression)
                field_key = "|".join(sorted(set(fields)))
                skeleton_key = "|".join(skeleton)

                dataset_id = str(raw.get("dataset_id") or "")
                category = _category_key(str(raw.get("category") or ""))
                if (not dataset_id or not category) and fields:
                    meta = self.field_catalog.meta(fields[0])
                    if meta:
                        dataset_id = dataset_id or meta.dataset_id
                        category = category or meta.category

                out.append(
                    {
                        "source_path": str(path),
                        "expression_hash": expression_hash,
                        "expression": expression,
                        "normalized_expression": expression,
                        "regime": regime,
                        "delay": delay,
                        "alpha_id": raw.get("alpha_id"),
                        "status": raw.get("status"),
                        "logged_at": raw.get("logged_at"),
                        "dataset_id": dataset_id,
                        "category": category,
                        "search_lane": raw.get("search_lane"),
                        "template_id": raw.get("template_id"),
                        "field_key": field_key,
                        "skeleton_key": skeleton_key,
                        "fields": fields,
                        "clusters": clusters,
                        "skeleton": skeleton,
                        "failed_checks": _failed_checks(raw),
                        "grade": raw.get("grade"),
                        "is_sharpe": raw.get("is_sharpe"),
                        "is_fitness": raw.get("is_fitness"),
                        "is_turnover": raw.get("is_turnover"),
                        "max_self_correlation": raw.get("max_self_correlation"),
                        "admission_decision": raw.get("admission_decision"),
                        "admission_reason": raw.get("admission_reason"),
                        "admission_global_max_corr": raw.get("admission_global_max_corr"),
                        "admission_brain_score_delta": raw.get("admission_brain_score_delta"),
                        "admission_delta_pool_sharpe": raw.get("admission_delta_pool_sharpe"),
                        "admission_signal": raw.get("admission_signal"),
                        "admission_fallback_delta_used": int(bool(raw.get("admission_fallback_delta_used"))),
                        "standalone_pass": int(_is_check_passing(raw)),
                    }
                )
        return out

    def _insert_records(self, conn: sqlite3.Connection, records: list[dict[str, Any]]) -> None:
        rows = []
        for r in records:
            rows.append(
                (
                    r["source_path"],
                    r["expression_hash"],
                    r["expression"],
                    r["normalized_expression"],
                    r["regime"],
                    r["delay"],
                    r["alpha_id"],
                    r["status"],
                    r["logged_at"],
                    r["dataset_id"],
                    r["category"],
                    r["search_lane"],
                    r["template_id"],
                    r["field_key"],
                    r["skeleton_key"],
                    _json_dumps(r["fields"]),
                    _json_dumps(r["clusters"]),
                    _json_dumps(r["skeleton"]),
                    _json_dumps(r["failed_checks"]),
                    r["grade"],
                    _safe_float_or_none(r["is_sharpe"]),
                    _safe_float_or_none(r["is_fitness"]),
                    _safe_float_or_none(r["is_turnover"]),
                    _safe_float_or_none(r["max_self_correlation"]),
                    r["admission_decision"],
                    r["admission_reason"],
                    _safe_float_or_none(r["admission_global_max_corr"]),
                    _safe_float_or_none(r["admission_brain_score_delta"]),
                    _safe_float_or_none(r["admission_delta_pool_sharpe"]),
                    r["admission_signal"],
                    int(bool(r["admission_fallback_delta_used"])),
                    r["standalone_pass"],
                )
            )
        conn.executemany(
            """
            INSERT INTO records (
                source_path, expression_hash, expression, normalized_expression, regime, delay,
                alpha_id, status, logged_at, dataset_id, category, search_lane, template_id,
                field_key, skeleton_key, fields_json, clusters_json, skeleton_json,
                failed_checks_json, grade, is_sharpe, is_fitness, is_turnover,
                max_self_correlation, admission_decision, admission_reason,
                admission_global_max_corr, admission_brain_score_delta,
                admission_delta_pool_sharpe, admission_signal,
                admission_fallback_delta_used, standalone_pass
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        conn.commit()

    def _build_summary(self, records: list[dict[str, Any]]) -> dict[str, Any]:
        dataset = Counter()
        category = Counter()
        field = Counter()
        cluster = Counter()
        operator = Counter()
        skeleton = Counter()
        failed_checks = Counter()
        admission = Counter()
        delay_counter = Counter()
        high_corr = Counter()
        pool_hurt: list[dict[str, Any]] = []
        pool_positive = 0
        standalone_pass = 0

        for r in records:
            dataset[r.get("dataset_id") or "unknown"] += 1
            category[r.get("category") or "unknown"] += 1
            field.update(r.get("fields") or [])
            cluster.update(r.get("clusters") or [])
            operator.update(r.get("skeleton") or [])
            if r.get("skeleton_key"):
                skeleton[r["skeleton_key"]] += 1
            failed_checks.update(r.get("failed_checks") or [])
            admission[r.get("admission_decision") or "UNSCORED"] += 1
            delay_counter[f"delay={r.get('delay', 1)}"] += 1
            if r.get("standalone_pass"):
                standalone_pass += 1

            corr = _safe_float_or_none(r.get("admission_global_max_corr"))
            if corr is None:
                corr = _safe_float_or_none(r.get("max_self_correlation"))
            if corr is not None and corr >= 0.70:
                for ds in [r.get("dataset_id") or "unknown"]:
                    high_corr[ds] += 1

            delta = _safe_float_or_none(r.get("admission_brain_score_delta"))
            if delta is None:
                delta = _safe_float_or_none(r.get("admission_delta_pool_sharpe"))
            if delta is not None:
                if delta > 0:
                    pool_positive += 1
                else:
                    pool_hurt.append(
                        {
                            "expression": r["expression"][:160],
                            "dataset_id": r.get("dataset_id"),
                            "fields": r.get("fields", [])[:4],
                            "skeleton": r.get("skeleton", [])[:8],
                            "contribution_delta": delta,
                            "signal": r.get("admission_signal"),
                            "global_corr": r.get("admission_global_max_corr"),
                        }
                    )

        pool_hurt.sort(key=lambda x: x["contribution_delta"])
        total = len(records)
        return {
            "total_records": total,
            "standalone_pass_records": standalone_pass,
            "dataset": _top_counter(dataset, 30),
            "category": _top_counter(category, 20),
            "field": _top_counter(field, 50),
            "cluster": _top_counter(cluster, 50),
            "operator": _top_counter(operator, 50),
            "skeleton": _top_counter(skeleton, 50),
            "failed_checks": _top_counter(failed_checks, 30),
            "admission": dict(admission),
            "delay": _top_counter(delay_counter, 5),
            "pool_impact": {
                "positive_delta": pool_positive,
                "non_positive_delta": len(pool_hurt),
                "worst_examples": pool_hurt[:30],
            },
            "high_corr_by_dataset": _top_counter(high_corr, 30),
            "source_paths": [str(p) for p in self.log_paths if p.exists()],
        }


def _safe_float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _top_counter(counter: Counter, n: int) -> list[dict[str, Any]]:
    return [{"key": k, "count": v} for k, v in counter.most_common(n)]


class HistoryIndex:
    """Query helper for a built history index."""

    def __init__(
        self,
        *,
        index_path: Path = DEFAULT_INDEX_PATH,
        summary_path: Path = DEFAULT_SUMMARY_PATH,
    ) -> None:
        self.index_path = Path(index_path)
        self.summary_path = Path(summary_path)
        self._semantic_cache: dict[int | None, set[str]] = {}

    def summary(self) -> dict[str, Any]:
        if not self.summary_path.exists():
            return {}
        return json.loads(self.summary_path.read_text())

    def is_compatible(self) -> bool:
        if not self.index_path.exists():
            return False
        conn = sqlite3.connect(self.index_path)
        try:
            cols = {row[1] for row in conn.execute("PRAGMA table_info(records)").fetchall()}
            return "delay" in cols
        except sqlite3.DatabaseError:
            return False
        finally:
            conn.close()

    def expression_exists(self, expression: str, regime: str | None = None) -> bool:
        if not self.index_path.exists():
            return False
        expr_hash = stable_hash(normalize_expression(expression), 20)
        conn = sqlite3.connect(self.index_path)
        try:
            if regime:
                row = conn.execute(
                    "SELECT 1 FROM records WHERE expression_hash=? AND regime=? LIMIT 1",
                    (expr_hash, regime),
                ).fetchone()
            else:
                row = conn.execute(
                    "SELECT 1 FROM records WHERE expression_hash=? LIMIT 1",
                    (expr_hash,),
                ).fetchone()
            return row is not None
        finally:
            conn.close()

    def expression_exists_for_delay(self, expression: str, target_delay: int) -> bool:
        if not self.index_path.exists():
            return False
        expr_hash = stable_hash(normalize_expression(expression), 20)
        conn = sqlite3.connect(self.index_path)
        try:
            row = conn.execute(
                "SELECT 1 FROM records WHERE expression_hash=? AND delay=? LIMIT 1",
                (expr_hash, int(target_delay)),
            ).fetchone()
            return row is not None
        finally:
            conn.close()

    def expression_exists_other_delay(self, expression: str, target_delay: int) -> bool:
        if not self.index_path.exists():
            return False
        expr_hash = stable_hash(normalize_expression(expression), 20)
        conn = sqlite3.connect(self.index_path)
        try:
            row = conn.execute(
                "SELECT 1 FROM records WHERE expression_hash=? AND delay<>? LIMIT 1",
                (expr_hash, int(target_delay)),
            ).fetchone()
            return row is not None
        finally:
            conn.close()

    def semantic_expression_exists(self, expression: str) -> bool:
        if not self.index_path.exists():
            return False
        return semantic_normalize_expression(expression) in self._semantic_expressions(None)

    def semantic_expression_exists_for_delay(self, expression: str, target_delay: int) -> bool:
        if not self.index_path.exists():
            return False
        return semantic_normalize_expression(expression) in self._semantic_expressions(int(target_delay))

    def semantic_expression_exists_other_delay(self, expression: str, target_delay: int) -> bool:
        if not self.index_path.exists():
            return False
        needle = semantic_normalize_expression(expression)
        conn = sqlite3.connect(self.index_path)
        try:
            rows = conn.execute(
                "SELECT expression FROM records WHERE delay<>?",
                (int(target_delay),),
            ).fetchall()
        finally:
            conn.close()
        return any(semantic_normalize_expression(row[0]) == needle for row in rows)

    def _semantic_expressions(self, target_delay: int | None) -> set[str]:
        if target_delay in self._semantic_cache:
            return self._semantic_cache[target_delay]
        conn = sqlite3.connect(self.index_path)
        try:
            if target_delay is None:
                rows = conn.execute("SELECT expression FROM records").fetchall()
            else:
                rows = conn.execute(
                    "SELECT expression FROM records WHERE delay=?",
                    (int(target_delay),),
                ).fetchall()
        finally:
            conn.close()
        values = {semantic_normalize_expression(row[0]) for row in rows if row and row[0]}
        self._semantic_cache[target_delay] = values
        return values

    def field_skeleton_exists(
        self,
        fields: Iterable[str],
        skeleton: Iterable[str],
        *,
        target_delay: int | None = None,
    ) -> bool:
        if not self.index_path.exists():
            return False
        field_key = "|".join(sorted(set(fields)))
        skeleton_key = "|".join(skeleton)
        conn = sqlite3.connect(self.index_path)
        try:
            if target_delay is None:
                row = conn.execute(
                    "SELECT 1 FROM records WHERE field_key=? AND skeleton_key=? LIMIT 1",
                    (field_key, skeleton_key),
                ).fetchone()
            else:
                row = conn.execute(
                    "SELECT 1 FROM records WHERE field_key=? AND skeleton_key=? AND delay=? LIMIT 1",
                    (field_key, skeleton_key, int(target_delay)),
                ).fetchone()
            return row is not None
        finally:
            conn.close()

    def hot_fingerprints(self, limit: int = 5000, *, target_delay: int | None = None) -> list[set[str]]:
        if not self.index_path.exists():
            return []
        conn = sqlite3.connect(self.index_path)
        try:
            if target_delay is None:
                rows = conn.execute(
                    """
                    SELECT fields_json, skeleton_json
                    FROM records
                    ORDER BY id DESC
                    LIMIT ?
                    """,
                    (limit,),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT fields_json, skeleton_json
                    FROM records
                    WHERE delay=?
                    ORDER BY id DESC
                    LIMIT ?
                    """,
                    (int(target_delay), limit),
                ).fetchall()
        finally:
            conn.close()
        fps: list[set[str]] = []
        for fields_json, skeleton_json in rows:
            fields = _json_loads(fields_json, [])
            skeleton = _json_loads(skeleton_json, [])
            fp = set(fields) | set(skeleton)
            if fp:
                fps.append(fp)
        return fps

    def jaccard_risk(
        self,
        fields: Iterable[str],
        skeleton: Iterable[str],
        threshold: float = 0.85,
        *,
        target_delay: int | None = None,
    ) -> dict[str, Any]:
        fp = set(fields) | set(skeleton)
        if not fp:
            return {"high": False, "max_jaccard": 0.0}
        max_j = 0.0
        for existing in self.hot_fingerprints(target_delay=target_delay):
            union = fp | existing
            j = len(fp & existing) / len(union) if union else 1.0
            max_j = max(max_j, j)
            if j >= threshold:
                return {"high": True, "max_jaccard": j}
        return {"high": False, "max_jaccard": max_j}

    def theory_risk_report(self, theory_text: str) -> dict[str, Any]:
        """Return simple text-overlap risks against history summaries."""
        summary = self.summary()
        text = theory_text.lower()
        risks: list[str] = []
        matches: dict[str, list[str]] = defaultdict(list)

        for kind in ("dataset", "category", "field", "cluster", "operator"):
            for row in summary.get(kind, [])[:30]:
                key = str(row.get("key") or "").lower()
                if key and key != "unknown" and key in text:
                    matches[kind].append(str(row["key"]))

        saturated_single_family = False
        cats = set(matches.get("category", []))
        dsets = set(matches.get("dataset", []))
        if cats and cats <= {"analyst", "fundamental"}:
            has_conditioning_family = any(
                term in text
                for term in ("news", "option", "volatility", "price", "volume", "social", "sentiment", "flow")
            )
            if not has_conditioning_family:
                saturated_single_family = True
                risks.append("theory is contained in already saturated analyst/fundamental families")
        if dsets and dsets <= {"analyst4", "fundamental6", "fundamental2"} and len(dsets) <= 1:
            risks.append("theory references only one historically crowded dataset")

        return {
            "matches": dict(matches),
            "risks": risks,
            "saturated_single_family": saturated_single_family,
        }
