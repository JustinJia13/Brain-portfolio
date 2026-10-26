"""Single home for all LLM prompts.

Each prompt is a markdown file in this folder with two sections:

    ## system
    <system prompt text>

    ## user
    <user message template with $placeholders>

Placeholders use string.Template syntax ($name / ${name}) so JSON braces in
the prompt body need no escaping. Load with:

    from wq_alpha.prompts import load_prompt
    p = load_prompt("theory_propose")
    system, user = p.system, p.render_user(cluster_catalog=..., n_theories=5)

Missing placeholders raise KeyError (substitute, not safe_substitute) so a
renamed variable fails loudly at call time instead of leaking "$name" into
the LLM context.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from string import Template

PROMPTS_DIR = Path(__file__).resolve().parent


@dataclass(frozen=True)
class Prompt:
    name: str
    system: str
    user_template: str

    def render_user(self, **kwargs: object) -> str:
        return Template(self.user_template).substitute(**{k: str(v) for k, v in kwargs.items()})


def load_prompt(name: str) -> Prompt:
    path = PROMPTS_DIR / f"{name}.md"
    text = path.read_text()
    sections: dict[str, list[str]] = {}
    current: str | None = None
    for line in text.splitlines():
        if line.strip().lower() in ("## system", "## user"):
            current = line.strip()[3:].lower()
            sections[current] = []
        elif current is not None:
            sections[current].append(line)
    if "system" not in sections or "user" not in sections:
        raise ValueError(f"prompt {name!r} must contain '## system' and '## user' sections")
    return Prompt(
        name=name,
        system="\n".join(sections["system"]).strip(),
        user_template="\n".join(sections["user"]).strip(),
    )


__all__ = ["Prompt", "load_prompt", "PROMPTS_DIR"]
