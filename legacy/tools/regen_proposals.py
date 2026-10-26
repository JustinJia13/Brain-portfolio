"""Disabled legacy combination proposal tool.

The active pipeline reviews and submits PROMOTE candidates individually rather
than combining multiple alphas into one expression.
"""

from __future__ import annotations


def main() -> None:
    print("Combination proposal generation is disabled.")
    print(
        "Use artifacts/full_scan/ranked.jsonl or "
        "artifacts/workshop/workshop_submittable.jsonl for individual review/submission."
    )


if __name__ == "__main__":
    main()
