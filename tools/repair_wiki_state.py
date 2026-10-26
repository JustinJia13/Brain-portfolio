"""One-shot repair of wiki/.state.json after mixed deterministic/LLM writes.

Scans every existing cluster page, detects whether its body is the
deterministic template or LLM narrative, and rewrites the state file with the
correct {digest, mode} per page based on current cluster_stats + mechanisms.
After this, `tools/build_wiki.py --llm` re-generates ONLY the pages that are
deterministic or stale, instead of re-paying for every page.

Usage (run after any in-flight build_wiki process has finished):
    python tools/repair_wiki_state.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from alpha_discovery.theory_method.context import load_clusters  # noqa: E402
from core.wiki.generate import (  # noqa: E402
    STATE_PATH,
    WIKI_DIR,
    _cluster_mechanisms,
    _input_hash,
    _load_mechanisms,
    _page_name,
    load_cluster_stats,
)


def looks_deterministic(body: str) -> bool:
    """The deterministic template always starts its body with '## Overview\nFields:'."""
    for line in body.splitlines():
        if line.startswith("## Overview"):
            idx = body.splitlines().index(line)
            rest = body.splitlines()[idx + 1: idx + 2]
            return bool(rest and rest[0].startswith("Fields:"))
    return False


def main() -> None:
    stats = load_cluster_stats()
    mechanisms = _load_mechanisms()
    load_clusters()  # validates catalog availability early

    pages: dict[str, dict] = {}
    n_llm = n_det = n_missing = 0
    for cid, row in stats.items():
        page_path = WIKI_DIR / "clusters" / f"{_page_name(cid)}.md"
        if not page_path.exists():
            n_missing += 1
            continue
        entries = _cluster_mechanisms(cid, mechanisms)
        digest = _input_hash(json.dumps(row, sort_keys=True),
                             json.dumps(entries, sort_keys=True, default=str))
        body = page_path.read_text()
        mode = "det" if looks_deterministic(body) else "llm"
        pages[cid] = {"digest": digest, "mode": mode}
        if mode == "llm":
            n_llm += 1
        else:
            n_det += 1

    STATE_PATH.write_text(json.dumps({"pages": pages}, indent=1))
    print(f"state rebuilt: {n_llm} llm pages, {n_det} deterministic pages, "
          f"{n_missing} clusters without pages")
    print("Next: python tools/build_wiki.py --llm   (regenerates only the deterministic/stale ones)")


if __name__ == "__main__":
    main()
