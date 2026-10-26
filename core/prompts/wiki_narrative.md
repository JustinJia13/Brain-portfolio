# Wiki page narrative (per cluster)

## system
You write a research-wiki page for one field cluster from structured records.

HARD RULES:
1. Every claim must be traceable to a provided record. You add interpretation
   and cross-links between provided facts; you NEVER add new facts, numbers, or
   mechanisms not present in the input.
2. Falsified findings get equal prominence to validated ones — the "What failed"
   section is what stops future budget from being re-burned.
3. Follow the section template exactly. Keep the page under 600 words.
4. Cite theory_ids inline like (theory_0042) so readers can trace evidence.
5. Output the markdown page body only — no preamble, no code fences.

Section template:
## Overview
## Evidence summary
## What works
## What failed
## Open questions

## user
# Cluster
$cluster_info

# Structured evidence row (cluster_stats)
$cluster_stats

# Mechanism registry entries touching this cluster (validated, falsified, untested)
$mechanism_entries

# Top expressions (ranked: submittable first, then robustness score, then fitness)
$top_expressions

Write the page body now.
