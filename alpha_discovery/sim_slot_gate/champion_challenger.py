"""champion_challenger.py — T2: new quality surrogate vs incumbent q-table, out-of-time.

Plan framing (PRE_DEPLOYMENT_VALIDATION.md Task 2):
  * The redundancy/dedup channel (c_max) is a PURE ADDITION (the q-table has no
    per-candidate pool dedup, confirmed) -> adopt regardless; quantified, not raced.
  * The head-to-head adjudicates ONLY the quality surrogate (Object B + UCB) vs the
    q-table's (field,op) bucket-mean pruning.
  * Scoring set = simulated alphas only (the sole rows with a realized label); the
    verdict is conditional on what the historical process chose to simulate.
  * Both policies built point-in-time on the OLDER split; evaluated on NEWER.

Outputs both frontiers, matched-recall gains, the disagreement-cell breakdown +
McNemar, and the 2e hypotheses (high-variance / sparse buckets).
"""

from __future__ import annotations

import json
import os
from collections import defaultdict
from datetime import datetime
from types import SimpleNamespace

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

from . import db_adapter as db
from .features import default_featurizer
from .gate import GateParams
from core.qtable import (
    compute_q_reward,
    extract_primary_op,
    MIN_OBS_L1,
    MIN_OBS_L2,
    MIN_PRUNE_SKELETONS,
)

_HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS = os.path.join(_HERE, "test_results")


# --------------------------------------------------------------------------- #
def _sim_result(rec: dict) -> SimpleNamespace:
    """Reconstruct the SimResult fields compute_q_reward reads (faithful to qtable)."""
    mc = rec.get("max_concentration")
    return SimpleNamespace(
        is_fitness=float(rec.get("is_fitness") or 0.0),
        is_sharpe=float(rec.get("is_sharpe") or 0.0),
        concentrated_weight_ok=bool(rec.get("is_concentrated_weight_ok", True)),
        sub_universe_sharpe_ok=bool(rec.get("is_sub_universe_sharpe_ok", True)),
        max_concentration=float(mc) if mc is not None else None,
        max_self_correlation=(float(rec["max_self_correlation"])
                              if rec.get("max_self_correlation") is not None else None),
        max_self_correlation_alpha_sharpe=(float(rec["max_self_correlation_alpha_sharpe"])
                              if rec.get("max_self_correlation_alpha_sharpe") is not None else None),
        yearly_stats=rec.get("yearly_stats") or None,
    )


class _Champion:
    """Point-in-time q-table bucket-mean pruner reconstructed from older sims."""

    def __init__(self):
        self.l1 = defaultdict(lambda: [0.0, 0, set()])   # (cluster,op) -> [sum,n,skeletons]
        self.l2 = defaultdict(lambda: [0.0, 0, set()])   # (field,op)   -> [sum,n,skeletons]

    def fit(self, recs: list[dict]):
        for rec in recs:
            fields = rec.get("fields") or []
            if not fields:
                continue
            field = str(fields[0])
            op = extract_primary_op(rec.get("expression", ""), field)
            if not op:
                continue
            try:
                reward = compute_q_reward(_sim_result(rec))
            except Exception:
                continue
            skel = str(rec.get("template_id") or "")
            clusters = rec.get("cluster_ids") or []
            cluster = str(clusters[0]) if clusters else ""
            for tbl, key in ((self.l2, (field, op)), (self.l1, (cluster, op))):
                e = tbl[key]; e[0] += reward; e[1] += 1; e[2].add(skel)
        return self

    def _prune(self, e, min_obs, theta):
        s, n, sk = e
        if n < min_obs or (s / n) >= theta:
            return False
        return len(sk) >= MIN_PRUNE_SKELETONS

    def skip(self, rec: dict, theta: float) -> bool:
        """Skip if the candidate's L1 or L2 bucket is 'dead' at threshold theta."""
        fields = rec.get("fields") or []
        if not fields:
            return False
        field = str(fields[0]); op = extract_primary_op(rec.get("expression", ""), field)
        if not op:
            return False
        clusters = rec.get("cluster_ids") or []; cluster = str(clusters[0]) if clusters else ""
        l2 = self.l2.get((field, op)); l1 = self.l1.get((cluster, op))
        if l2 and self._prune(l2, MIN_OBS_L2, theta):
            return True
        if l1 and self._prune(l1, MIN_OBS_L1, theta):
            return True
        return False


# --------------------------------------------------------------------------- #
def run(n_test_cap: int = 8000, seed: int = 0):
    fz = default_featurizer()  # pruned phi (post Task 1)
    recs, t, phis, passes = [], [], [], []
    for d in db.iter_master():
        e = d.get("expression"); ts = d.get("logged_at")
        if not isinstance(e, str) or not ts or d.get("is_sharpe") is None:
            continue
        if d.get("is_fitness") is None or d.get("is_turnover") is None:
            continue
        phi = fz.featurize(e)
        if phi is None:
            continue
        try:
            to = datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp()
        except ValueError:
            continue
        recs.append(d); t.append(to); phis.append(phi)
        passes.append((d["is_sharpe"] >= 1.25) and (d["is_fitness"] >= 1.0) and (d["is_turnover"] <= 0.7))
    t = np.array(t); X = np.array(phis); passes = np.array(passes, bool)
    order = np.argsort(t); split = t[order[int(len(t) * 0.7)]]
    tr = t <= split; te = t > split
    print(f"n={len(recs)} train={tr.sum()} test={te.sum()} base_good={passes.mean():.3f}")

    # ---- champion: q-table from OLDER split ---------------------------------
    champ = _Champion().fit([recs[i] for i in np.where(tr)[0]])

    # ---- challenger: pruned quality surrogate from OLDER split ---------------
    sh = np.array([r["is_sharpe"] for r in recs], float)
    rt = np.array([r.get("is_returns") if r.get("is_returns") is not None else np.nan for r in recs], float)
    tu = np.array([r["is_turnover"] for r in recs], float)
    ss = np.array([r.get("check_low_sub_universe_sharpe_value") if r.get("check_low_sub_universe_sharpe_value") is not None else np.nan for r in recs], float)

    def fitq(y, q):
        m = tr & np.isfinite(y)
        return HistGradientBoostingRegressor(loss="quantile", quantile=q, max_iter=250,
            max_leaf_nodes=31, min_samples_leaf=50, learning_rate=0.08,
            early_stopping=True, random_state=0).fit(X[m], y[m])
    heads = {name: {q: fitq(y, q) for q in (0.16, 0.5, 0.84)}
             for name, y in (("subsharpe", ss), ("sharpe", sh), ("ret", rt), ("turnover", tu))
             if np.isfinite(y[tr]).sum() >= 60}

    teix = np.where(te)[0]
    rng = np.random.default_rng(seed)
    if len(teix) > n_test_cap:
        teix = teix[rng.permutation(len(teix))[:n_test_cap]]
    Xte = X[teix]; yte = passes[teix]
    p = GateParams()

    def hq(name, q):
        return heads[name][q].predict(Xte) if name in heads else np.full(len(Xte), np.nan)
    sb50, sb84 = hq("subsharpe", 0.5), hq("subsharpe", 0.84)
    sh50, sh84 = hq("sharpe", 0.5), hq("sharpe", 0.84)
    rt50, rt84 = hq("ret", 0.5), hq("ret", 0.84)
    to50, to84 = hq("turnover", 0.5), hq("turnover", 0.84)

    def challenger_skip(beta):
        s_ucb = sb50 + beta * np.abs(sb84 - sb50)
        fit_hi = (sh50 + beta * np.abs(sh84 - sh50)) * np.sqrt(
            np.abs(rt50 + beta * np.abs(rt84 - rt50)) / np.maximum(to50 - beta * np.abs(to84 - to50), 0.125))
        to_lcb = to50 - 1.0 * np.abs(to84 - to50)
        sub_ok = ~np.isfinite(s_ucb) | (s_ucb >= p.tau_sub)
        fit_ok = ~np.isfinite(fit_hi) | (fit_hi >= p.tau_fit)
        kill_to = np.isfinite(to_lcb) & (to_lcb > p.tau_to)
        return (~(sub_ok & fit_ok)) | kill_to     # skip mask (quality only; no c_max)

    def frontier_of(skipfn, grid):
        out = []
        for g in grid:
            skip = skipfn(g)
            kept = ~skip
            recall = float((yte & kept).sum() / max(yte.sum(), 1))
            saved = float(skip.mean())
            bad_skipped = float((skip & ~yte).sum() / max((~yte).sum(), 1))
            out.append({"param": float(g), "recall": recall, "slots_saved": saved, "bad_skipped": bad_skipped, "skip": skip})
        return out

    champ_front = frontier_of(lambda th: np.array([champ.skip(recs[i], th) for i in teix]),
                              grid=[0.02, 0.05, 0.08, 0.12, 0.18, 0.25, 0.35, 0.5])
    chal_front = frontier_of(challenger_skip, grid=[1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0])

    print("\n=== CHAMPION (q-table) frontier ===")
    for r in champ_front: print(f"  theta={r['param']:.2f}  recall={r['recall']:.3f}  slots_saved={r['slots_saved']:.3f}  bad_skipped={r['bad_skipped']:.3f}")
    print("=== CHALLENGER (pruned quality surrogate) frontier ===")
    for r in chal_front: print(f"  beta={r['param']:.1f}  recall={r['recall']:.3f}  slots_saved={r['slots_saved']:.3f}  bad_skipped={r['bad_skipped']:.3f}")

    # ---- matched-recall comparison + McNemar on disagreements ---------------
    def at_recall(front, target):
        return min(front, key=lambda r: abs(r["recall"] - target))
    print("\n=== MATCHED-RECALL gains + disagreement (McNemar) ===")
    mn = {}
    for target in (0.85, 0.90):
        c = at_recall(champ_front, target); h = at_recall(chal_front, target)
        sc, sh_ = c["skip"], h["skip"]
        # 'correct' decision = keep good or skip bad
        champ_correct = (sc & ~yte) | (~sc & yte)
        chal_correct = (sh_ & ~yte) | (~sh_ & yte)
        b = int((champ_correct & ~chal_correct).sum())   # champ right, chal wrong
        cc = int((~champ_correct & chal_correct).sum())   # chal right, champ wrong
        chi = (abs(b - cc) - 1) ** 2 / (b + cc) if (b + cc) > 0 else 0.0
        # disagreement breakdown on good vs bad
        dis = sc != sh_
        good_dis = dis & yte; bad_dis = dis & ~yte
        chal_keeps_good = int((good_dis & ~sh_).sum()); champ_keeps_good = int((good_dis & ~sc).sum())
        chal_skips_bad = int((bad_dis & sh_).sum()); champ_skips_bad = int((bad_dis & sc).sum())
        mn[target] = {"champ_saved": c["slots_saved"], "chal_saved": h["slots_saved"],
                      "champ_recall": c["recall"], "chal_recall": h["recall"],
                      "mcnemar_b_champ_only": b, "mcnemar_c_chal_only": cc, "mcnemar_chi2": chi,
                      "good_disagree_chal_keeps": chal_keeps_good, "good_disagree_champ_keeps": champ_keeps_good,
                      "bad_disagree_chal_skips": chal_skips_bad, "bad_disagree_champ_skips": champ_skips_bad}
        print(f"  @recall~{target}: champ saves {c['slots_saved']:.3f} (r={c['recall']:.3f}) vs "
              f"chal saves {h['slots_saved']:.3f} (r={h['recall']:.3f})  Δsaved={h['slots_saved']-c['slots_saved']:+.3f}")
        print(f"     disagreements: good kept by chal/champ={chal_keeps_good}/{champ_keeps_good}  "
              f"bad skipped by chal/champ={chal_skips_bad}/{champ_skips_bad}  McNemar χ²={chi:.2f} (chal-only={cc},champ-only={b})")

    # ---- 2e hypothesis: high-variance & sparse buckets ----------------------
    print("\n=== 2e: where the surrogate should win (bucket variance / sparsity) ===")
    bvar = {}
    for i in np.where(tr)[0]:
        fields = recs[i].get("fields") or []
        if not fields: continue
        f = str(fields[0]); op = extract_primary_op(recs[i].get("expression",""), f)
        if not op: continue
        try: r = compute_q_reward(_sim_result(recs[i]))
        except Exception: continue
        bvar.setdefault((f,op),[]).append(r)
    hv = {k for k,v in bvar.items() if len(v)>=8 and np.mean(v)<0.3 and np.std(v)>0.5}  # low mean, high var
    sparse = {k for k,v in bvar.items() if len(v)<MIN_OBS_L2}
    def keyof(i):
        fields=recs[i].get("fields") or []
        if not fields: return None
        f=str(fields[0]); return (f, extract_primary_op(recs[i].get("expression",""),f))
    c_at = at_recall(champ_front,0.85); h_at = at_recall(chal_front,0.85)
    for label, cells in (("high-variance",hv),("sparse",sparse)):
        idx=[j for j,i in enumerate(teix) if keyof(i) in cells]
        if not idx:
            print(f"  {label}: no test alphas"); continue
        idx=np.array(idx); good=yte[idx]
        cr=float((good & ~c_at['skip'][idx]).sum()/max(good.sum(),1))
        hr=float((good & ~h_at['skip'][idx]).sum()/max(good.sum(),1))
        print(f"  {label} buckets: n_test={len(idx)} good={int(good.sum())}  good-recall champ={cr:.3f} vs chal={hr:.3f}")
        mn[f"hyp_{label}"]={"n_test":len(idx),"good":int(good.sum()),"champ_recall":cr,"chal_recall":hr}

    out={"n":len(recs),"base_good":float(passes.mean()),
         "champion_frontier":[{k:v for k,v in r.items() if k!='skip'} for r in champ_front],
         "challenger_frontier":[{k:v for k,v in r.items() if k!='skip'} for r in chal_front],
         "matched":mn}
    os.makedirs(RESULTS,exist_ok=True)
    json.dump(out,open(os.path.join(RESULTS,"champion_challenger.json"),"w"),indent=2,default=float)
    print("\nsaved test_results/champion_challenger.json")
    return out


if __name__ == "__main__":
    run()
