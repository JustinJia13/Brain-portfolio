"""Embedding-only mechanism dedup with LLM adjudication (no taxonomy).

Identity resolution for the mechanism registry: a re-proposed mechanism is a
lookup that inherits the existing posterior, not an error — and a falsified
mechanism must not re-enter under new wording. Flow per new card:

    embed(hypothesis + clusters + horizon)  ->  top-k cosine vs registry
      max cosine < EMBED_NEW_THRESHOLD  ->  NEW (no LLM call)
      otherwise -> adjudicator (prompts/theory_adjudicate) rules
        SAME(id)        inherit posterior; falsified -> reject
        VARIANT(id, Δ)  new cell, discounted parent prior
        NEW             register fresh

Rulings are stored on the registry record so a pair is never re-adjudicated.
Embeddings live in a sidecar .npz (never in the JSONL).
"""

from __future__ import annotations

import fcntl
import json
import math
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from core import llm as anthropic
from core.json_utils import parse_json_object
from core.prompts import load_prompt

REGISTRY_DIR = Path("artifacts/theory")
REGISTRY_PATH = REGISTRY_DIR / "mechanisms.jsonl"
EMBEDDINGS_PATH = REGISTRY_DIR / "mechanism_embeddings.npz"
LANE_LOCK_PATH = REGISTRY_DIR / ".state.lock"


@contextmanager
def lane_lock():
    """Exclusive lock over the shared theory state (registry + master + wiki).

    Held only for the brief per-round state commit, never during simulation,
    so brain2 and brain discovery can run concurrently and only serialize their
    quick read-modify-write of shared files.
    """
    REGISTRY_DIR.mkdir(parents=True, exist_ok=True)
    with open(LANE_LOCK_PATH, "w") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)

EMBED_MODEL = "text-embedding-3-small"
EMBED_NEW_THRESHOLD = 0.60
ADJUDICATE_TOP_K = 5
VARIANT_PRIOR_DISCOUNT = 0.5  # fraction of parent Beta pseudo-counts inherited


def card_embed_text(card: dict) -> str:
    return " | ".join([
        str(card.get("hypothesis") or ""),
        str(card.get("why_it_exists") or ""),
        "clusters: " + ",".join(card.get("cluster_ids") or []),
        f"horizon: {card.get('horizon_band')}",
        f"direction: {card.get('direction')}",
    ])


def embed_texts(texts: list[str], api_key: str | None = None) -> list[list[float]]:
    import openai

    # Embeddings stay on OpenAI — Anthropic has no embeddings endpoint — so this
    # must use the OpenAI key, not the (now Anthropic) default chat key.
    client = openai.OpenAI(api_key=anthropic.load_openai_key(api_key))
    resp = client.embeddings.create(model=EMBED_MODEL, input=texts)
    return [d.embedding for d in resp.data]


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


@dataclass
class Ruling:
    ruling: str               # NEW | SAME | VARIANT
    matched_id: str | None
    delta: str | None
    reason: str
    max_cosine: float

    @property
    def rejected(self) -> bool:
        return self.ruling == "REJECTED"


class MechanismRegistry:
    """mechanisms.jsonl + embedding sidecar, with dedup against new cards.

    Concurrency-safe for simultaneous brain2 + brain discovery: theory IDs are
    account-namespaced (no cross-process collisions) and save() merges with the
    current on-disk state so one writer never drops another's mechanisms. The
    caller must hold lane_lock() around save() so the read-modify-write is
    atomic across processes.
    """

    def __init__(self, path: Path = REGISTRY_PATH, embeddings_path: Path = EMBEDDINGS_PATH,
                 *, account: str | None = None) -> None:
        self.path = path
        self.embeddings_path = embeddings_path
        self.account = account
        self.cards: dict[str, dict] = {}
        self.embeddings: dict[str, list[float]] = {}
        self.cards = self._read_disk_cards()
        self.embeddings = self._read_disk_embeddings()

    def _read_disk_cards(self) -> dict[str, dict]:
        cards: dict[str, dict] = {}
        if self.path.exists():
            with self.path.open() as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        card = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    tid = card.get("theory_id")
                    if tid:
                        cards[tid] = card  # last line wins (latest update)
        return cards

    def _read_disk_embeddings(self) -> dict[str, list[float]]:
        if self.embeddings_path.exists():
            import numpy as np

            data = np.load(self.embeddings_path)
            return {k: data[k].tolist() for k in data.files}
        return {}

    def save(self) -> None:
        """Merge in-memory state with disk, then write. Hold lane_lock()."""
        disk_cards = self._read_disk_cards()
        disk_cards.update(self.cards)   # this process's versions win for its own ids
        self.cards = disk_cards         # adopt others' cards added since load
        disk_emb = self._read_disk_embeddings()
        disk_emb.update(self.embeddings)
        self.embeddings = disk_emb

        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".jsonl.tmp")
        with tmp.open("w") as f:
            for card in self.cards.values():
                f.write(json.dumps(card) + "\n")
        tmp.replace(self.path)          # atomic swap
        if self.embeddings:
            import numpy as np

            np.savez_compressed(
                self.embeddings_path,
                **{k: np.asarray(v, dtype="float32") for k, v in self.embeddings.items()},
            )

    def reload(self) -> None:
        """Re-read disk so dedup/index see another process's committed cards.

        Safe to call at a round boundary (no unsaved in-memory cards then).
        """
        self.cards = self._read_disk_cards()
        self.embeddings = self._read_disk_embeddings()

    def next_theory_id(self) -> str:
        prefix = f"{self.account}_theory_" if self.account else "theory_"
        n = 0
        while f"{prefix}{n:04d}" in self.cards:
            n += 1
        return f"{prefix}{n:04d}"

    def top_matches(self, embedding: list[float], k: int = ADJUDICATE_TOP_K) -> list[tuple[str, float]]:
        scored = [
            (tid, cosine(embedding, emb))
            for tid, emb in self.embeddings.items()
            if tid in self.cards
        ]
        scored.sort(key=lambda t: t[1], reverse=True)
        return scored[:k]

    def index_lines(self) -> str:
        """Compact one-line-per-mechanism index for the proposer context."""
        lines = []
        for tid, card in self.cards.items():
            lines.append(
                f"{tid} | {card.get('verdict') or 'UNTESTED'} | "
                f"support={card.get('last_support_rate', 'n/a')} | "
                f"{str(card.get('hypothesis') or '')[:110]}"
            )
        return "\n".join(lines) if lines else "(registry empty)"

    def falsified_lines(self) -> str:
        lines = [
            f"{tid}: {str(card.get('hypothesis') or '')[:110]}"
            f" — {str(card.get('falsification_reason') or 'family evidence')[:90]}"
            for tid, card in self.cards.items()
            if card.get("verdict") == "FALSIFIED"
        ]
        return "\n".join(lines) if lines else "(none yet)"

    # ── dedup ────────────────────────────────────────────────────────────
    def adjudicate(
        self,
        card: dict,
        embedding: list[float],
        *,
        client: "anthropic.Anthropic | None" = None,
        model: str = "gpt-5.5",
    ) -> Ruling:
        matches = self.top_matches(embedding)
        max_cos = matches[0][1] if matches else 0.0
        if max_cos < EMBED_NEW_THRESHOLD:
            return Ruling("NEW", None, None, f"max cosine {max_cos:.2f} below threshold", max_cos)

        matched_blocks = []
        for tid, score in matches:
            existing = self.cards[tid]
            matched_blocks.append(json.dumps({
                "theory_id": tid,
                "cosine": round(score, 3),
                "hypothesis": existing.get("hypothesis"),
                "why_it_exists": existing.get("why_it_exists"),
                "cluster_ids": existing.get("cluster_ids"),
                "horizon_band": existing.get("horizon_band"),
                "verdict": existing.get("verdict") or "UNTESTED",
                "falsification_reason": existing.get("falsification_reason"),
            }))
        prompt = load_prompt("theory_adjudicate")
        client = client or anthropic.Anthropic()
        response = client.messages.create(
            model=model,
            max_completion_tokens=600,
            system=prompt.system,
            messages=[{"role": "user", "content": prompt.render_user(
                new_card=json.dumps({k: card.get(k) for k in (
                    "hypothesis", "why_it_exists", "cluster_ids", "direction",
                    "horizon_band", "operator_families")}),
                matched_cards="\n".join(matched_blocks),
            )}],
        )
        parsed = parse_json_object(response.content[0].text) or {}
        ruling = str(parsed.get("ruling") or "NEW").upper()
        matched_id = parsed.get("matched_id")
        delta = parsed.get("delta")
        reason = str(parsed.get("reason") or "")

        if ruling == "SAME" and matched_id in self.cards:
            if self.cards[matched_id].get("verdict") == "FALSIFIED":
                return Ruling("REJECTED", matched_id, None,
                              f"same as falsified {matched_id}: {reason}", max_cos)
            return Ruling("SAME", matched_id, None, reason, max_cos)
        if ruling == "VARIANT" and matched_id in self.cards:
            return Ruling("VARIANT", matched_id, str(delta or ""), reason, max_cos)
        return Ruling("NEW", None, None, reason or "adjudicated new", max_cos)

    def register(self, card: dict, embedding: list[float], ruling: Ruling) -> dict:
        """Apply a ruling: register NEW/VARIANT cards, return the canonical card."""
        if ruling.ruling == "SAME" and ruling.matched_id:
            canonical = self.cards[ruling.matched_id]
            canonical.setdefault("adjudications", []).append(
                {"against": card.get("hypothesis"), "ruling": "SAME", "reason": ruling.reason})
            return canonical

        tid = card.get("theory_id") or self.next_theory_id()
        card = dict(card)
        card["theory_id"] = tid
        card.setdefault("beta_alpha", 1.0)
        card.setdefault("beta_beta", 1.0)
        if ruling.ruling == "VARIANT" and ruling.matched_id:
            parent = self.cards[ruling.matched_id]
            card["variant_of"] = ruling.matched_id
            card["variant_delta"] = ruling.delta
            card["beta_alpha"] = 1.0 + VARIANT_PRIOR_DISCOUNT * (float(parent.get("beta_alpha") or 1.0) - 1.0)
            card["beta_beta"] = 1.0 + VARIANT_PRIOR_DISCOUNT * (float(parent.get("beta_beta") or 1.0) - 1.0)
        self.cards[tid] = card
        self.embeddings[tid] = embedding
        return card


__all__ = [
    "EMBED_NEW_THRESHOLD",
    "MechanismRegistry",
    "Ruling",
    "card_embed_text",
    "cosine",
    "embed_texts",
]
