"""Exploration memory: structured facts distilled from batch simulation results.

ExplorationMemory stores up to max_facts MemoryFact objects, persisted as JSON.
MemoryDistiller uses an LLM to extract new facts from a completed batch and
update the existing fact set.
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path

from wq_alpha import llm as anthropic

_LOG = logging.getLogger(__name__)

_DISTILLER_SYSTEM = """\
You are a memory distiller for a WorldQuant Brain alpha search system.

After each batch of alpha simulations, you maintain a compact fact base (max 50 facts)
that captures reusable patterns about which operator/field combinations work or fail.

Your tasks each round:
1. Extract new insights from the latest batch results
2. Confirm or update existing facts that gained new supporting evidence
3. Remove or downgrade stale/unsupported facts
4. Output the complete updated fact list as a JSON array

Fact types:
- "operator_insight": which operator category or combination works/fails well
- "field_insight": a specific field or field cluster that has strong/weak signal
- "structure_insight": a grammar-level pattern (budget K, nesting, group/CS combo)
- "failure_pattern": systematic check failure mode to avoid (e.g. CONCENTRATED_WEIGHT)

Confidence guidelines:
- 0.9+: strong evidence from multiple passing alphas with high fitness
- 0.7-0.9: moderate evidence (2+ supporting examples)
- 0.5-0.7: weak/early evidence (1 example or ambiguous signal)
- below 0.5: stale or refuted — drop these

Output ONLY a valid JSON array of fact objects. No prose, no markdown, just raw JSON.
Each fact must have these exact keys:
  fact_id, type, confidence, observation, supporting_expressions,
  operators, fields, budget_range, fitness_evidence, created_batch, last_confirmed_batch
"""


def _fmt_grade_dist(records: list[dict]) -> str:
    grades: dict[str, int] = {}
    for r in records:
        g = (r.get("grade") or "UNKNOWN").upper()
        grades[g] = grades.get(g, 0) + 1
    return "  " + ", ".join(f"{g}={n}" for g, n in sorted(grades.items()))


def _fmt_record_line(r: dict) -> str:
    sharpe = r.get("is_sharpe")
    fitness = r.get("is_fitness")
    expr = r.get("expression", "")
    grade = (r.get("grade") or "?").upper()
    fields = r.get("fields") or []
    checks = r.get("checks") or []
    failing = [c["name"] for c in checks if c.get("result") == "FAIL"]
    s = f"  [{sharpe:.3f}sh, {fitness:.3f}fit, {grade}]"
    if failing:
        s += f" FAIL({', '.join(failing)})"
    s += f"  {expr[:80]}"
    if fields:
        s += f"  fields={fields[:2]}"
    return s


@dataclass
class MemoryFact:
    """One reusable pattern extracted from alpha simulation results."""
    fact_id: str
    type: str  # operator_insight | field_insight | structure_insight | failure_pattern
    confidence: float
    observation: str
    supporting_expressions: list[str] = field(default_factory=list)
    operators: list[str] = field(default_factory=list)
    fields: list[str] = field(default_factory=list)
    budget_range: list[int] = field(default_factory=list)
    fitness_evidence: float = 0.0
    created_batch: int = 0
    last_confirmed_batch: int = 0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "MemoryFact":
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in d.items() if k in known})


class ExplorationMemory:
    """Persistent store of MemoryFact objects.

    Loads from / saves to a JSON file.  Trimmed to max_facts by confidence.
    """

    def __init__(self, path: Path | None = None, max_facts: int = 50) -> None:
        self._path = path or Path(__file__).parent / "artifacts" / "memory.json"
        self._max_facts = max_facts
        self._facts: list[MemoryFact] = []
        self.load()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def load(self) -> None:
        if not self._path.exists():
            return
        try:
            with self._path.open() as f:
                data = json.load(f)
            self._facts = [MemoryFact.from_dict(d) for d in data]
            _LOG.info("Loaded %d memory facts from %s", len(self._facts), self._path)
        except Exception as exc:
            _LOG.warning("Failed to load memory from %s: %s — starting empty", self._path, exc)
            self._facts = []

    def save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_suffix(".tmp")
        try:
            with tmp.open("w") as f:
                json.dump([fact.to_dict() for fact in self._facts], f, indent=2)
            tmp.replace(self._path)
        except Exception as exc:
            _LOG.error("Failed to save memory to %s: %s", self._path, exc)

    # ------------------------------------------------------------------
    # Access
    # ------------------------------------------------------------------

    @property
    def facts(self) -> list[MemoryFact]:
        return list(self._facts)

    def context_for_proposer(self, top_n: int = 20) -> list[dict]:
        """Return top-N highest-confidence facts as plain dicts for prompt injection."""
        sorted_facts = sorted(self._facts, key=lambda f: f.confidence, reverse=True)
        return [f.to_dict() for f in sorted_facts[:top_n]]

    def update(self, new_facts: list[MemoryFact]) -> None:
        """Replace fact set with new_facts, trim to max_facts by confidence, save."""
        sorted_facts = sorted(new_facts, key=lambda f: f.confidence, reverse=True)
        self._facts = sorted_facts[: self._max_facts]
        self.save()


class MemoryDistiller:
    """Uses an LLM to distill batch simulation results into MemoryFact objects.

    One Claude API call per round.  On any LLM/parse failure, returns the
    existing facts unchanged (never corrupts memory).
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "claude-opus-4-6",
    ) -> None:
        resolved_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        if not resolved_key:
            raise ValueError(
                "No OpenAI API key.\n"
                "Set OPENAI_API_KEY env var or pass api_key= to MemoryDistiller."
            )
        self.client = anthropic.Anthropic(api_key=resolved_key)
        self.model = model

    def distill(
        self,
        batch_records: list[dict],
        current_facts: list[MemoryFact],
        batch_index: int,
        admission_results: dict[str, dict] | None = None,
    ) -> list[MemoryFact]:
        """Update and return fact list based on the latest batch results.

        admission_results: optional dict mapping expression → admission metrics.
            Brain score delta is preferred; fallback delta_pool_sharpe is used
            only when Brain comparison was unavailable.

        Returns current_facts unchanged on any failure.
        """
        if not batch_records:
            return current_facts

        done = [r for r in batch_records if r.get("status") == "done"]
        if not done:
            return current_facts

        user_msg = self._build_user_message(done, current_facts, batch_index, admission_results)
        try:
            response = self.client.messages.create(
                model=self.model,
                max_completion_tokens=8192,
                system=_DISTILLER_SYSTEM,
                messages=[{"role": "user", "content": user_msg}],
            )
            raw = response.content[0].text.strip()
            return self._parse_response(raw, current_facts, batch_index)
        except Exception as exc:
            _LOG.warning("MemoryDistiller LLM call failed: %s — keeping existing facts", exc)
            return current_facts

    def _build_user_message(
        self,
        done_records: list[dict],
        current_facts: list[MemoryFact],
        batch_index: int,
        admission_results: dict[str, dict] | None = None,
    ) -> str:
        by_sharpe = sorted(done_records, key=lambda r: float(r.get("is_sharpe") or 0.0), reverse=True)

        # Top-10 by Sharpe — annotate with pool impact when admission data is available
        top10_lines = []
        for r in by_sharpe[:10]:
            line = _fmt_record_line(r)
            if admission_results is not None:
                expr = r.get("expression", "")
                metrics = admission_results.get(expr)
                if metrics:
                    delta = metrics.get("brain_score_delta")
                    if delta is None:
                        delta = metrics.get("delta_pool_sharpe")
                    if delta is not None:
                        sign = "+" if delta > 0 else ""
                        tag = "helps pool" if delta > 0 else "HURTS POOL"
                        line += f"  → contribution_delta={sign}{delta:.4f}  {tag}"
                    else:
                        line += "  → pool score: not scored"
                else:
                    line += "  → pool score: not scored yet"
            top10_lines.append(line)

        # Near-misses: good Sharpe + failing checks
        near_misses = [
            r for r in by_sharpe
            if float(r.get("is_sharpe") or 0.0) >= 1.0
            and (r.get("grade") or "").upper() != "PASS"
        ]
        nm_lines = [_fmt_record_line(r) for r in near_misses[:5]]

        # Bottom-5 (weakest signal)
        bottom5_lines = [_fmt_record_line(r) for r in by_sharpe[-5:]]

        grade_dist = _fmt_grade_dist(done_records)
        current_facts_json = json.dumps([f.to_dict() for f in current_facts], indent=2)

        top_label = f"TOP {min(10, len(by_sharpe))} BY SHARPE"
        if admission_results is not None:
            top_label += " (with pool impact — do NOT praise alphas marked HURTS POOL)"

        parts = [
            f"Batch #{batch_index} results ({len(done_records)} completed expressions):",
            "",
            f"GRADE DISTRIBUTION:\n{grade_dist}",
            "",
            f"{top_label}:",
            "\n".join(top10_lines) if top10_lines else "  (none)",
        ]
        if nm_lines:
            parts += ["", "NEAR-MISSES (good Sharpe, failing checks):", "\n".join(nm_lines)]
        parts += [
            "",
            "WEAKEST RESULTS:",
            "\n".join(bottom5_lines) if bottom5_lines else "  (none)",
            "",
            f"CURRENT FACTS ({len(current_facts)} facts):",
            current_facts_json,
            "",
            "Update the facts based on this batch. Keep max 50. Output updated JSON array:",
        ]
        return "\n".join(parts)

    def _parse_response(
        self,
        raw: str,
        current_facts: list[MemoryFact],
        batch_index: int,
    ) -> list[MemoryFact]:
        import re

        # Strip markdown fences if present (handles ```json ... ``` blocks)
        text = raw
        if text.startswith("```"):
            lines = text.splitlines()
            text = "\n".join(
                line for line in lines
                if not line.startswith("```")
            )

        data = None
        # Direct parse first
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            pass

        # Fallback: find the JSON array (handles truncated responses or leading prose)
        if data is None:
            m = re.search(r"\[.*", text, re.DOTALL)
            if m:
                fragment = m.group(0)
                # If response was truncated mid-JSON, try to salvage complete objects
                try:
                    data = json.loads(fragment)
                except json.JSONDecodeError:
                    # Strip trailing incomplete object and close the array
                    last_close = fragment.rfind("}")
                    if last_close != -1:
                        trimmed = fragment[: last_close + 1] + "\n]"
                        try:
                            data = json.loads(trimmed)
                            _LOG.info("MemoryDistiller: recovered %d facts from truncated response", len(data))
                        except json.JSONDecodeError:
                            pass

        if data is None:
            _LOG.warning("MemoryDistiller: could not parse JSON response — keeping existing facts")
            return current_facts

        if not isinstance(data, list):
            _LOG.warning("MemoryDistiller: expected list, got %s — keeping existing facts", type(data))
            return current_facts

        facts: list[MemoryFact] = []
        for item in data:
            if not isinstance(item, dict):
                continue
            # Ensure fact_id
            if not item.get("fact_id"):
                item["fact_id"] = f"f_{uuid.uuid4().hex[:8]}"
            # Ensure created_batch
            if not item.get("created_batch"):
                item["created_batch"] = batch_index
            try:
                facts.append(MemoryFact.from_dict(item))
            except Exception as exc:
                _LOG.debug("Skipping malformed fact: %s — %s", item, exc)
                continue

        if not facts:
            _LOG.warning("MemoryDistiller: LLM returned 0 parseable facts — keeping existing")
            return current_facts

        _LOG.info("MemoryDistiller: extracted %d facts from batch %d", len(facts), batch_index)
        return facts


    def bootstrap_from_history(
        self,
        sim_records: list[dict],
        memory: "ExplorationMemory",
        total_sample: int = 5000,
        chunk_size: int = 100,
    ) -> list[MemoryFact]:
        """Seed memory from historical sim records via multi-pass chunked distillation.

        Samples up to `total_sample` records with a failure-heavy ratio, then
        runs distill() in chunks of `chunk_size`, carrying facts forward each
        pass so the full history is reflected.  Only called when memory is empty.

        Returns the seeded fact list (also written to memory).
        """
        import random

        done = [r for r in sim_records if r.get("status") == "done"]
        if not done:
            _LOG.info("bootstrap_from_history: no done records — skipping")
            return []

        # Partition into three tiers
        truly_passing = [
            r for r in done
            if float(r.get("is_sharpe") or 0.0) >= 1.25
            and float(r.get("is_fitness") or 0.0) >= 1.0
            and not any(c.get("result") == "FAIL" for c in (r.get("checks") or []))
        ]
        near_misses = [
            r for r in done
            if float(r.get("is_sharpe") or 0.0) >= 1.0
            and any(c.get("result") == "FAIL" for c in (r.get("checks") or []))
        ]
        failures = [
            r for r in done
            if any(c.get("result") == "FAIL" for c in (r.get("checks") or []))
            and float(r.get("is_sharpe") or 0.0) < 1.0
        ]

        # Target ratio: 60% failures, 30% near-misses, 10% passing
        n_fail = min(int(total_sample * 0.60), len(failures))
        n_near = min(int(total_sample * 0.30), len(near_misses))
        n_pass = min(total_sample - n_fail - n_near, len(truly_passing))

        # Spread failures evenly across all records to cover diverse operators/check types
        f_step = max(1, len(failures) // n_fail) if n_fail else 1
        fail_sample = failures[::f_step][:n_fail]

        nm_step = max(1, len(near_misses) // n_near) if n_near else 1
        near_sample = near_misses[::nm_step][:n_near]

        pass_sample = truly_passing[:n_pass]

        # Interleave: failures first, then near-misses, then passing
        combined: list[dict] = []
        seen: set[str] = set()
        for r in fail_sample + near_sample + pass_sample:
            expr = r.get("expression", "")
            if expr not in seen:
                seen.add(expr)
                combined.append(r)

        # Shuffle so each chunk gets a mix of outcomes
        random.shuffle(combined)

        _LOG.info(
            "bootstrap_from_history: %d records total (failures=%d near_miss=%d passing=%d) → %d chunks",
            len(combined), len(fail_sample), len(near_sample), len(pass_sample),
            (len(combined) + chunk_size - 1) // chunk_size,
        )

        facts: list[MemoryFact] = []
        for i in range(0, len(combined), chunk_size):
            chunk = combined[i : i + chunk_size]
            batch_idx = i // chunk_size
            _LOG.info("bootstrap_from_history: chunk %d/%d (%d records, %d current facts)",
                      batch_idx + 1, (len(combined) + chunk_size - 1) // chunk_size,
                      len(chunk), len(facts))
            facts = self.distill(chunk, facts, batch_index=batch_idx)

        if facts:
            memory.update(facts)
            _LOG.info("bootstrap_from_history: wrote %d facts to memory after %d chunks",
                      len(facts), (len(combined) + chunk_size - 1) // chunk_size)
        return facts


__all__ = ["MemoryFact", "ExplorationMemory", "MemoryDistiller"]
