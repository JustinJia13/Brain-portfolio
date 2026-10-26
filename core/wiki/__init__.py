"""Research wiki: the persistent posterior store of the theory loop.

Two layers with a load-bearing separation:
- structured layer (wiki/structured/*.jsonl): machine-written only, from sim
  records and the mechanism registry. Authoritative. Drives budget allocation
  and the grammar prior export.
- narrative layer (wiki/clusters/*.md etc.): LLM-distilled from the structured
  layer only — interpretation and cross-links, never new facts.
"""

from core.wiki.generate import build_cluster_stats, export_priors, generate_wiki

__all__ = ["build_cluster_stats", "export_priors", "generate_wiki"]
