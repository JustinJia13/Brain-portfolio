"""LLM-driven alpha exploration module.

Replaces random Q-table sampling with an LLM-guided loop:
  1. AlphaProposerAgent: LLM picks GrammarConfig patch → grammar samples valid FASTEXPR
  2. SimilarityFilter: drops near-duplicates vs submitted history
  3. ExplorationMemory + MemoryDistiller: LLM distills results into persistent facts

Usage:
    python -m alpha_discovery/legacy.llm_grammar_control_method.runner --rounds 5 --batch-size 40
"""

from .memory import ExplorationMemory, MemoryDistiller, MemoryFact
from .proposer import AlphaProposerAgent
from .similarity import SimilarityFilter

__all__ = [
    "AlphaProposerAgent",
    "SimilarityFilter",
    "ExplorationMemory",
    "MemoryDistiller",
    "MemoryFact",
    "LLMExplorationLoop",
]


def __getattr__(name: str):
    if name == "LLMExplorationLoop":
        from .runner import LLMExplorationLoop

        return LLMExplorationLoop
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
