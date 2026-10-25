#!/usr/bin/env python3
"""Lab-grown question classes for Jason's framework, with per-instance
ground-truth pi via branch-resampling.

Pipeline:
 1. Sample candidate states: beta ~ near-critical band, run to day 19.
 2. For each candidate, branch M times (deepcopy + reseed dists), run to day 90;
    pi = fraction of branches where EVENT occurs. Event: cumulative cases
    exceed THRESH_MULT x day-19 cumulative cases by day 60.
 3. Certify + select two classes:
    SKILL class: bimodal pi (aim mean ~0.10, high Var(pi))
    CALIB class: pi in a narrow low band (irreducible by construction)
 4. Report every framework quantity: E[pi(1-pi)], Var(pi), resolvable share,
    climatology Brier, perfect-forecaster Brier, and a prefix-feature logistic
    baseline's CV-R2 against pi (can the visible prefix recover pi?).
Outputs corpus JSON (question text + context + pi) ready for model eval.
"""
import json, sys, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

N_CANDIDATES = 420
M_BRANCH = 100
SNAP = 19          # branch/report day (history identical through this index)
RES_DAY = 60
THRESH_MULT = 9.0  # event: cum cases by day 60 >= 9x cum cases at day 19
BETA_LO, BETA_HI = 0.006, 0.026   # R0 ~ 0.36..1.56 — mostly sub/near-critical
OUT = Path("/private/tmp/claude-501/-Users-elsehow-Projects-s-tier/f2b46dcd-fe73-45c4-8c11-533351736920/scratchpad/lab_classes.json")

def candidate(args):
    """Run one candidate to day 19, branch M times, return pi + prefix series."""
    import starsim as ss
    import sciris as sc
    idx, beta, seed = args
    def make():
        sir = ss.SIR(init_prev=0.005, beta=beta, dur_inf=10, p_death=0.02)
        return ss.Sim(dict(n_agents=5000, networks=dict(type="random", n_contacts=6),
                           diseases=sir, dur=90, dt=1, verbose=0, rand_seed=seed))
    s = make()
    s.init()
    for _ in range(SNAP + 1):
        s.run_one_step()
    base = np.array(s.results["sir"]["cum_infections"])[:SNAP + 1]
    active = np.array(s.results["sir"]["n_infected"])[:SNAP + 1]
    cum19 = float(base[SNAP])
    thresh = THRESH_MULT * max(cum19, 25.0)
    hits = 0
    for b in range(M_BRANCH):
        c = sc.dcp(s)
        rs = 100_000 + idx * 1_000 + b
        c.pars.rand_seed = rs
        c.dists.init(obj=c, base_seed=rs, force=True)
        c.run()
        cum = np.array(c.results["sir"]["cum_infections"])
        if float(cum[RES_DAY]) >= thresh:
            hits += 1
    pi = hits / M_BRANCH
    return {
        "idx": idx, "beta": beta, "seed": seed, "pi": pi,
        "cum19": cum19, "thresh": thresh,
        "prefix_cum": base.tolist(), "prefix_active": active.tolist(),
    }

def main():
    t0 = time.time()
    rng = np.random.default_rng(23)
    jobs = [(i, float(rng.uniform(BETA_LO, BETA_HI)), int(rng.integers(1, 60_000)))
            for i in range(N_CANDIDATES)]
    out = []
    with ProcessPoolExecutor(max_workers=8) as ex:
        for i, r in enumerate(ex.map(candidate, jobs, chunksize=2)):
            out.append(r)
            if (i + 1) % 20 == 0:
                print(f"{i+1}/{N_CANDIDATES} candidates ({time.time()-t0:.0f}s)", flush=True)
    pis = np.array([r["pi"] for r in out])
    print(f"\ncandidate pi distribution: mean={pis.mean():.3f} "
          f"deciles={np.round(np.percentile(pis,[10,25,50,75,90]),3).tolist()}")

    def certify(sel, label):
        p = np.array([r["pi"] for r in sel])
        irre = float(np.mean(p * (1 - p)))         # E[pi(1-pi)]
        varp = float(p.var())                       # Var(pi) = resolvable variance
        clim = irre + varp                          # flat-forecast Brier
        share = varp / clim if clim else 0.0
        # can the visible prefix recover pi? ridge on prefix features, 5-fold CV R2
        X = []
        for r in sel:
            c = np.array(r["prefix_cum"]); a = np.array(r["prefix_active"])
            g1 = (c[SNAP] + 1) / (c[SNAP - 7] + 1)          # week growth
            g2 = (c[SNAP - 7] + 1) / (c[SNAP - 14] + 1)
            X.append([c[SNAP], a[SNAP], g1, g2, g1 / (g2 + 1e-9), a[SNAP] / (c[SNAP] + 1)])
        X = np.array(X); y = p
        Xs = (X - X.mean(0)) / (X.std(0) + 1e-9)
        n = len(y); fold = np.arange(n) % 5
        pred = np.zeros(n)
        for f in range(5):
            tr, te = fold != f, fold == f
            A = Xs[tr]; lam = 1.0
            w = np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ (y[tr] - y[tr].mean()))
            pred[te] = np.clip(y[tr].mean() + Xs[te] @ w, 0, 1)
        cvr2 = 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()
        print(f"\n[{label}] n={len(sel)}  mean pi={p.mean():.3f}")
        print(f"  E[pi(1-pi)] (irreducible)      = {irre:.4f}   <- perfect-forecaster Brier")
        print(f"  Var(pi)     (resolvable)       = {varp:.4f}")
        print(f"  climatology Brier              = {clim:.4f}")
        print(f"  resolvable share (headroom)    = {share:.0%}")
        print(f"  prefix-feature CV-R2 vs pi     = {cvr2:.2f}  (is the signal visible?)")
        return dict(n=len(sel), mean_pi=float(p.mean()), irreducible=irre,
                    resolvable=varp, climatology=clim, share=share, cv_r2=float(cvr2))

    # SKILL class: bimodal tail — low-pi majority + genuinely contingent minority
    lo = [r for r in out if r["pi"] <= 0.03]
    hi = [r for r in out if 0.15 <= r["pi"] <= 0.75]
    k = min(len(hi), max(8, len(hi)))
    rng2 = np.random.default_rng(5)
    n_lo = min(len(lo), 4 * k)
    skill = list(rng2.choice(lo, n_lo, replace=False)) + list(hi)
    # CALIB class: narrow low band — irreducible by construction
    calib = [r for r in out if 0.02 <= r["pi"] <= 0.10]

    stats = {"skill": certify(skill, "SKILL class"),
             "calib": certify(calib, "CALIBRATION class")}

    corpus = {"meta": {"snap": SNAP, "res_day": RES_DAY, "thresh_mult": THRESH_MULT,
                       "m_branch": M_BRANCH, "beta_band": [BETA_LO, BETA_HI],
                       "stats": stats},
              "skill": [dict(r, kind="skill") for r in skill],
              "calib": [dict(r, kind="calib") for r in calib]}
    OUT.write_text(json.dumps(corpus))
    print(f"\n-> {OUT}  ({time.time()-t0:.0f}s total)")

if __name__ == "__main__":
    main()
