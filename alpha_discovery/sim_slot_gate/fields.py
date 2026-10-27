"""fields.py — open-vocabulary field embedding for the field-similarity channel (§3.3).

~10^4 fields exist; we never build the full field x field matrix.  Each field gets a
low-rank unit vector v_f in R^d seeded from metadata (dataset / category /
subcategory / units / hashed description tokens).  Similarity is the dot product
<v_f, v_g>.  A never-simulated field is usable immediately because the seed needs
only metadata (or, failing that, the tokens of the field id itself).

Refinement in PnL space from single-field probe alphas (spec §3.3 v2) is deferred;
``FieldEmbedder.refine`` is left as a documented hook.

Determinism: feature hashing uses md5 (Python's builtin hash() is salted per run).
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

_HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_META_JSON = os.path.join(_HERE, "datafield_metadata.json")
_FIELDS_CSV = os.path.join(_HERE, "all_fields.csv")

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STOP = frozenset(
    "the a an of for and or to in on by per with is are total value last "
    "data company equity number net".split()
)


@dataclass(frozen=True)
class _FieldMeta:
    dataset: str = ""
    category: str = ""
    subcategory: str = ""
    ftype: str = ""
    description: str = ""


def _stable_bucket_sign(token: str, d: int) -> tuple[int, float]:
    h = hashlib.md5(token.encode()).digest()
    bucket = int.from_bytes(h[:4], "little") % d
    sign = 1.0 if (h[4] & 1) else -1.0
    return bucket, sign


def _tokens(text: str) -> list[str]:
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOP and len(t) > 1]


def _load_metadata() -> dict[str, _FieldMeta]:
    meta: dict[str, _FieldMeta] = {}
    # broad set from CSV first
    if os.path.exists(_FIELDS_CSV):
        with open(_FIELDS_CSV, newline="") as fh:
            for row in csv.DictReader(fh):
                fid = (row.get("field_id") or "").strip()
                if not fid:
                    continue
                meta[fid] = _FieldMeta(
                    dataset=(row.get("dataset_id") or "").strip(),
                    category=(row.get("category") or "").strip(),
                    subcategory=(row.get("subcategory") or "").strip(),
                    ftype=(row.get("type") or "").strip(),
                    description=(row.get("field_name") or "").strip(),
                )
    # richer descriptions from the JSON override
    if os.path.exists(_META_JSON):
        with open(_META_JSON) as fh:
            try:
                data = json.load(fh)
            except Exception:
                data = []
        for rec in data if isinstance(data, list) else []:
            fid = (rec.get("id") or "").strip()
            if not fid:
                continue
            ds = (rec.get("dataset") or {})
            cat = (rec.get("category") or {})
            sub = (rec.get("subcategory") or {})
            meta[fid] = _FieldMeta(
                dataset=(ds.get("id") if isinstance(ds, dict) else str(ds)) or "",
                category=(cat.get("id") if isinstance(cat, dict) else str(cat)) or "",
                subcategory=(sub.get("id") if isinstance(sub, dict) else str(sub)) or "",
                ftype=(rec.get("type") or ""),
                description=(rec.get("description") or ""),
            )
    return meta


class FieldEmbedder:
    """Metadata-seeded field embedding with a lazy per-field cache."""

    # relative weights: structured metadata dominates, description is secondary
    W_DATASET = 1.6
    W_CATEGORY = 1.4
    W_SUBCAT = 1.0
    W_TYPE = 0.4
    W_DESC = 0.5          # per description token
    W_IDTOK = 0.6         # per field-id token (fallback signal, always added)

    def __init__(self, d: int = 48, metadata: dict[str, _FieldMeta] | None = None):
        self.d = d
        self.meta = metadata if metadata is not None else _load_metadata()
        self._cache: dict[str, np.ndarray] = {}

    # ---- embedding -------------------------------------------------------- #
    def _accumulate(self, vec: np.ndarray, token: str, weight: float) -> None:
        b, s = _stable_bucket_sign(token, self.d)
        vec[b] += weight * s

    def embed(self, field_id: str) -> np.ndarray:
        v = self._cache.get(field_id)
        if v is not None:
            return v
        vec = np.zeros(self.d, dtype=float)
        m = self.meta.get(field_id)
        if m is not None:
            if m.dataset:
                self._accumulate(vec, f"ds={m.dataset}", self.W_DATASET)
            if m.category:
                self._accumulate(vec, f"cat={m.category}", self.W_CATEGORY)
            if m.subcategory:
                self._accumulate(vec, f"sub={m.subcategory}", self.W_SUBCAT)
            if m.ftype:
                self._accumulate(vec, f"ty={m.ftype}", self.W_TYPE)
            for tok in _tokens(m.description):
                self._accumulate(vec, f"w={tok}", self.W_DESC)
        # always fold in the field-id's own tokens (handles engineered/derived
        # fields absent from metadata, e.g. *_derivative composites)
        for tok in _tokens(field_id):
            self._accumulate(vec, f"id={tok}", self.W_IDTOK)
        n = np.linalg.norm(vec)
        if n > 1e-12:
            vec /= n
        self._cache[field_id] = vec
        return vec

    def rho_field(self, f: str, g: str) -> float:
        if f == g:
            return 1.0
        return float(np.clip(np.dot(self.embed(f), self.embed(g)), -1.0, 1.0))

    # ---- set-to-set similarity (§3.3) ------------------------------------ #
    def field_sim(self, A: list[str], B: list[str]) -> float:
        """Symmetric best-match set similarity. Empty-vs-empty -> 1.0."""
        A = [a for a in A if a]
        B = [b for b in B if b]
        if not A and not B:
            return 1.0
        if not A or not B:
            return 0.0
        VA = [self.embed(a) for a in A]
        VB = [self.embed(b) for b in B]
        M = np.array([[float(np.dot(va, vb)) for vb in VB] for va in VA])  # |A| x |B|
        a2b = M.max(axis=1).mean()
        b2a = M.max(axis=0).mean()
        return float(np.clip(0.5 * (a2b + b2a), -1.0, 1.0))

    # ---- deferred refinement hook (spec §3.3 v2) ------------------------- #
    def refine(self, probe_pnl_corr: dict) -> None:  # pragma: no cover
        """Placeholder: refine v_f from single-field probe-alpha PnL correlations.
        Not implemented in v1 (metadata seed only). Documented in metric.md §9."""
        raise NotImplementedError("PnL-space field refinement is a v2 feature")


@lru_cache(maxsize=1)
def default_embedder() -> FieldEmbedder:
    return FieldEmbedder()
