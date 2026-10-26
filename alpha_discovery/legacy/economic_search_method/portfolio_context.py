"""Submitted-alpha portfolio context for economic hypothesis search."""

from __future__ import annotations

import json
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from core.brain.api import sign_in

from .history_index import DEFAULT_FIELD_CSV, FieldCatalog
from .schemas import normalize_expression, stable_hash


PACKAGE_DIR = Path(__file__).resolve().parent
DEFAULT_ARTIFACTS_DIR = PACKAGE_DIR / "artifacts"
DEFAULT_PORTFOLIO_CONTEXT_PATH = DEFAULT_ARTIFACTS_DIR / "current_portfolio_context.json"
DEFAULT_COMPETITION_ID = "IQC2026S1"

BRAIN_BASE = "https://api.worldquantbrain.com"
_OP_RE = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\(")


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _regular_code(alpha: dict[str, Any]) -> str:
    regular = alpha.get("regular") or {}
    if isinstance(regular, dict):
        return str(regular.get("code") or "")
    return str(regular or "")


def _compact_settings(settings: dict[str, Any]) -> dict[str, Any]:
    return {
        "delay": settings.get("delay"),
        "decay": settings.get("decay"),
        "neutralization": settings.get("neutralization"),
        "universe": settings.get("universe"),
    }


def _regime_key(settings: dict[str, Any]) -> str:
    return (
        f"universe={settings.get('universe')};"
        f"delay={settings.get('delay')};"
        f"decay={settings.get('decay')};"
        f"neutralization={settings.get('neutralization')}"
    )


def _is_submitted(alpha: dict[str, Any]) -> bool:
    status = str(alpha.get("status") or "").upper()
    stage = str(alpha.get("stage") or "").upper()
    if alpha.get("dateSubmitted"):
        return True
    if status in {"SUBMITTED", "ACTIVE", "LIVE", "PROD"}:
        return True
    if stage in {"OS", "PROD", "SUBMITTED"}:
        return True
    competitions = alpha.get("competitions") or []
    return bool(competitions)


def _paged_get(
    session,
    endpoint: str,
    *,
    params: dict[str, Any],
    limit: int,
    max_pages: int,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    offset = 0
    page_no = 0
    while True:
        page_no += 1
        if max_pages and page_no > max_pages:
            break
        resp = session.get(
            f"{BRAIN_BASE}{endpoint}",
            params={**params, "limit": limit, "offset": offset},
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        results = list(data.get("results") or [])
        out.extend(results)
        total = int(data.get("count") or len(out))
        if not results or len(out) >= total or len(results) < limit:
            break
        offset += limit
    return out


def _fetch_competition_alphas(
    session,
    *,
    competition_id: str,
    limit: int,
    max_pages: int,
) -> list[dict[str, Any]]:
    if not competition_id:
        return []
    return _paged_get(
        session,
        f"/competitions/{competition_id}/alphas",
        params={},
        limit=limit,
        max_pages=max_pages,
    )


def _fetch_account_submitted_scan(
    session,
    *,
    limit: int,
    max_pages: int,
) -> list[dict[str, Any]]:
    submitted: list[dict[str, Any]] = []
    try:
        submitted = _paged_get(
            session,
            "/users/self/alphas",
            params={"status": "SUBMITTED"},
            limit=limit,
            max_pages=max_pages,
        )
    except Exception:
        submitted = []

    if submitted:
        return submitted

    raw = _paged_get(session, "/users/self/alphas", params={}, limit=limit, max_pages=max_pages)
    return [alpha for alpha in raw if _is_submitted(alpha)]


def fetch_submitted_alpha_details(
    *,
    credentials: str = "brain.txt",
    competition_id: str = DEFAULT_COMPETITION_ID,
    alpha_ids: Iterable[str] | None = None,
    include_account_scan: bool = False,
    limit: int = 100,
    max_pages: int = 0,
    sleep_seconds: float = 0.1,
) -> list[dict[str, Any]]:
    """Fetch all submitted alphas visible under the authenticated account.

    Brain's current submitted portfolio is exposed by competition endpoint, not
    by ``/users/self/alphas?status=SUBMITTED`` for this account. The competition
    endpoint is therefore the primary source. An optional account scan can be
    enabled to merge any submitted records found under ``/users/self/alphas``.
    """
    session = sign_in(credentials)

    explicit_ids = [str(alpha_id).strip() for alpha_id in (alpha_ids or []) if str(alpha_id).strip()]
    if explicit_ids:
        submitted: list[dict[str, Any]] = [{"id": alpha_id, "status": "SUBMITTED"} for alpha_id in explicit_ids]
    else:
        submitted = _fetch_competition_alphas(
            session,
            competition_id=competition_id,
            limit=limit,
            max_pages=max_pages,
        )
        if include_account_scan:
            submitted.extend(_fetch_account_submitted_scan(session, limit=limit, max_pages=max_pages))

        try:
            if not submitted:
                submitted = _fetch_account_submitted_scan(session, limit=limit, max_pages=max_pages)
        except Exception:
            submitted = []

    details: list[dict[str, Any]] = []
    seen: set[str] = set()
    for alpha in submitted:
        alpha_id = str(alpha.get("id") or alpha.get("alpha") or "")
        if not alpha_id or alpha_id in seen:
            continue
        seen.add(alpha_id)
        try:
            resp = session.get(f"{BRAIN_BASE}/alphas/{alpha_id}", timeout=30)
            if resp.status_code == 200:
                detail = resp.json()
            else:
                detail = alpha
        except Exception:
            detail = alpha
        if not _is_submitted(detail) and not _is_submitted(alpha):
            continue
        details.append({**alpha, **detail})
        if sleep_seconds > 0:
            time.sleep(sleep_seconds)
    return details


def build_portfolio_context(
    alphas: list[dict[str, Any]],
    *,
    field_csv: Path = DEFAULT_FIELD_CSV,
) -> dict[str, Any]:
    catalog = FieldCatalog(field_csv)
    submitted: list[dict[str, Any]] = []
    theme_counter: Counter = Counter()
    theme_fields: dict[str, Counter] = defaultdict(Counter)
    theme_ops: dict[str, Counter] = defaultdict(Counter)
    dataset_counter: Counter = Counter()
    category_counter: Counter = Counter()
    field_counter: Counter = Counter()
    operator_counter: Counter = Counter()
    setting_counter: Counter = Counter()
    regime_counter: Counter = Counter()
    delay_counter: Counter = Counter()

    for alpha in alphas:
        expression = normalize_expression(_regular_code(alpha))
        fields = catalog.extract_fields(expression)
        metas = [catalog.meta(f) for f in fields]
        datasets = sorted({m.dataset_id for m in metas if m is not None})
        categories = sorted({m.category for m in metas if m is not None})
        operators = _OP_RE.findall(expression)
        settings = alpha.get("settings") or {}
        compact_settings = _compact_settings(settings)
        is_stats = alpha.get("is") or {}
        theme = _theme(categories, datasets, fields, operators)
        delay_value = compact_settings.get("delay")

        dataset_counter.update(datasets or ["unknown"])
        category_counter.update(categories or ["unknown"])
        field_counter.update(fields)
        operator_counter.update(operators)
        theme_counter[theme] += 1
        theme_fields[theme].update(fields)
        theme_ops[theme].update(operators)
        regime_counter[_regime_key(compact_settings)] += 1
        if delay_value is not None:
            delay_counter[f"delay={delay_value}"] += 1
        for key, value in compact_settings.items():
            if value is not None:
                setting_counter[f"{key}={value}"] += 1

        submitted.append(
            {
                "alpha_id": alpha.get("id") or alpha.get("alpha"),
                "expression_hash": stable_hash(expression, 20),
                "expression": expression,
                "fields": fields,
                "datasets": datasets,
                "categories": categories,
                "operators": operators,
                "theme": theme,
                "sharpe": _safe_float(is_stats.get("sharpe")),
                "fitness": _safe_float(is_stats.get("fitness")),
                "turnover": _safe_float(is_stats.get("turnover")),
                "grade": alpha.get("grade"),
                "date_submitted": alpha.get("dateSubmitted"),
                "status": alpha.get("status"),
                "stage": alpha.get("stage"),
                "settings": compact_settings,
            }
        )

    theme_exposures = []
    for theme, count in theme_counter.most_common():
        theme_exposures.append(
            {
                "theme": theme,
                "count": count,
                "top_fields": [k for k, _ in theme_fields[theme].most_common(8)],
                "top_operators": [k for k, _ in theme_ops[theme].most_common(8)],
                "avoid_reason": "already submitted portfolio exposure; require clear orthogonal conditioning",
            }
        )

    do_not_repeat = []
    for exposure in theme_exposures:
        if exposure["count"] >= 1:
            do_not_repeat.append(exposure["theme"])
    for field, count in field_counter.most_common(12):
        if count >= 1:
            do_not_repeat.append(f"submitted field exposure: {field}")

    return {
        "source": "brain_account_submitted_alphas",
        "n_submitted": len(submitted),
        "submitted_alphas": submitted,
        "theme_exposures": theme_exposures,
        "dataset_exposures": [{"key": k, "count": v} for k, v in dataset_counter.most_common()],
        "category_exposures": [{"key": k, "count": v} for k, v in category_counter.most_common()],
        "field_exposures": [{"key": k, "count": v} for k, v in field_counter.most_common(50)],
        "operator_exposures": [{"key": k, "count": v} for k, v in operator_counter.most_common(50)],
        "delay_exposures": [{"key": k, "count": v} for k, v in delay_counter.most_common()],
        "setting_exposures": [{"key": k, "count": v} for k, v in setting_counter.most_common()],
        "regime_exposures": [{"key": k, "count": v} for k, v in regime_counter.most_common()],
        "do_not_repeat": do_not_repeat[:40],
    }


def fetch_and_write_portfolio_context(
    *,
    credentials: str = "brain.txt",
    competition_id: str = DEFAULT_COMPETITION_ID,
    alpha_ids: Iterable[str] | None = None,
    include_account_scan: bool = False,
    out_path: Path = DEFAULT_PORTFOLIO_CONTEXT_PATH,
    field_csv: Path = DEFAULT_FIELD_CSV,
    limit: int = 100,
    max_pages: int = 0,
    sleep_seconds: float = 0.1,
) -> dict[str, Any]:
    alphas = fetch_submitted_alpha_details(
        credentials=credentials,
        competition_id=competition_id,
        alpha_ids=alpha_ids,
        include_account_scan=include_account_scan,
        limit=limit,
        max_pages=max_pages,
        sleep_seconds=sleep_seconds,
    )
    context = build_portfolio_context(alphas, field_csv=field_csv)
    context["competition_id"] = competition_id
    context["explicit_alpha_ids"] = [
        str(alpha_id).strip() for alpha_id in (alpha_ids or []) if str(alpha_id).strip()
    ]
    context["include_account_scan"] = include_account_scan
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(context, indent=2, sort_keys=True))
    return context


def load_portfolio_context(path: Path = DEFAULT_PORTFOLIO_CONTEXT_PATH) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return {}


def compact_portfolio_context(context: dict[str, Any], *, target_delay: int | None = None) -> dict[str, Any]:
    if not context:
        return {}
    submitted = list(context.get("submitted_alphas") or [])
    same_delay = [
        a for a in submitted if target_delay is not None and _alpha_delay(a) == int(target_delay)
    ]
    delay_exposures = context.get("delay_exposures") or _delay_exposures_from_submitted(submitted)
    return {
        "n_submitted": context.get("n_submitted", 0),
        "target_delay": target_delay,
        "theme_exposures": (context.get("theme_exposures") or [])[:20],
        "dataset_exposures": (context.get("dataset_exposures") or [])[:20],
        "category_exposures": (context.get("category_exposures") or [])[:20],
        "field_exposures": (context.get("field_exposures") or [])[:30],
        "operator_exposures": (context.get("operator_exposures") or [])[:30],
        "delay_exposures": delay_exposures[:10],
        "setting_exposures": (context.get("setting_exposures") or [])[:30],
        "regime_exposures": (context.get("regime_exposures") or [])[:20],
        "do_not_repeat": (context.get("do_not_repeat") or [])[:40],
        "submitted_alphas": [_compact_alpha(a) for a in submitted[:40]],
        "same_delay_submitted_alphas": [_compact_alpha(a) for a in same_delay[:30]],
    }


def _alpha_delay(alpha: dict[str, Any]) -> int | None:
    settings = alpha.get("settings") or {}
    try:
        return int(settings.get("delay"))
    except (TypeError, ValueError):
        return None


def _delay_exposures_from_submitted(submitted: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counter: Counter = Counter()
    for alpha in submitted:
        delay = _alpha_delay(alpha)
        if delay is not None:
            counter[f"delay={delay}"] += 1
    return [{"key": k, "count": v} for k, v in counter.most_common()]


def _compact_alpha(alpha: dict[str, Any]) -> dict[str, Any]:
    return {
        "alpha_id": alpha.get("alpha_id"),
        "theme": alpha.get("theme"),
        "fields": alpha.get("fields", [])[:6],
        "datasets": alpha.get("datasets", [])[:4],
        "operators": alpha.get("operators", [])[:8],
        "sharpe": alpha.get("sharpe"),
        "fitness": alpha.get("fitness"),
        "grade": alpha.get("grade"),
        "settings": alpha.get("settings") or {},
        "expression": str(alpha.get("expression") or "")[:180],
    }


def _theme(categories: list[str], datasets: list[str], fields: list[str], operators: list[str]) -> str:
    fam = "+".join(categories or datasets or ["unknown"])
    top_fields = "+".join(fields[:3]) if fields else "no_fields"
    top_ops = ">".join(operators[:3]) if operators else "raw"
    return f"{fam} | {top_fields} | {top_ops}"
