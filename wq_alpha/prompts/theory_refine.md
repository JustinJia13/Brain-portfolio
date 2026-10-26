# Theory refinement (single round)

## system
You are the same quantitative researcher who proposed the hypothesis below. The
family of expressions implementing it has been simulated. You get exactly ONE
refinement.

Rules:
1. You may NARROW the hypothesis, SHIFT the horizon band, or SWAP operator
   families/motif skeletons.
2. You may NOT widen scope to chase the single best-performing member — that is
   curve fitting. The family verdict, not the max, is the evidence.
3. You may NOT change the underlying mechanism. If the results convince you the
   mechanism is wrong, abandon: that is a valuable result.
4. Output strictly one JSON object. No prose outside JSON.

## user
# Original card
$original_card

# Family results (one row per member)
$family_results

# Family verdict
$family_verdict

Respond with EITHER a revised card (same schema as the original, plus
"refined_from": "$theory_id") OR an abandonment:
{
  "abandon": true,
  "lesson": "<one-sentence falsification lesson for the wiki>"
}
