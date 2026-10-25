"""Does a weights-accessible LLM beat a simple statistical baseline on ForecastBench?

The existing reanalysis studies LLMs relative to each other (ECI × Brier) and excludes the
baselines. This asks the question those omit: on the SAME resolved binary questions, does
Llama-3.3-70B (the weights-accessible model closest to what we mech-interp) beat:
  - the FB Naive Forecaster (market price for markets; no-change-ish for datasets),
  - the FB Imputed Forecaster,
  - Always-0.5,
  - a leave-one-out CLIMATOLOGY (base rate of "1" by source × horizon bin) — the proper
    statistical baseline, analogous to the sim climatology that beat the 70B on FreeCiv.
Split by source group (dataset vs market), source, and horizon.

    cd /Users/elsehow/Projects/llm-forecasting
    uv run python experiments/forecastbench-reanalysis/baserate_vs_llm.py
"""
import json
import os
from collections import defaultdict
from datetime import datetime

import numpy as np

BASE = "/Users/elsehow/Projects/forecastbench-datasets/processed_forecast_sets"
DATASET_SRC = {"yfinance", "fred", "dbnomics", "acled", "wikipedia"}
MARKET_SRC = {"manifold", "metaculus", "polymarket"}
HBINS = [(0, 10, "1wk"), (11, 45, "1mo"), (46, 135, "3mo"), (136, 270, "6mo"), (271, 9999, "12mo+")]
FOCUS = ["Llama-3.3-70B-Instruct-Turbo (zero shot)", "Meta-Llama-3.1-405B-Instruct-Turbo (zero shot)",
         "Llama-3-70b-Chat-Hf (zero shot)"]


def hbin(d):
    if d is None:
        return "?"
    for lo, hi, lab in HBINS:
        if lo <= d <= hi:
            return lab
    return "?"


def brier(f, o):
    return float(np.mean((np.array(f) - np.array(o)) ** 2)) if len(f) else float("nan")


def load(base):
    llm = defaultdict(dict)        # model display name -> {qkey: forecast}
    base_fc = defaultdict(dict)    # baseline name -> {qkey: forecast}
    outcome, meta = {}, {}         # qkey -> outcome ; qkey -> (source, hbin)
    for dd in sorted(os.listdir(base)):
        dp = os.path.join(base, dd)
        if not os.path.isdir(dp):
            continue
        for f in os.listdir(dp):
            if not f.endswith(".json"):
                continue
            fl = f.lower()
            bname = ("naive" if "naive-forecaster" in fl else "imputed" if "imputed-forecaster" in fl
                     else "always0.5" if "always-0.5" in fl else None)
            is_llm = ("_zero_shot.json" in f) and ("with_freeze_values" not in f)
            if bname is None and not is_llm:
                continue
            try:
                data = json.load(open(os.path.join(dp, f)))
                due = data["forecast_due_date"]
                due_dt = datetime.fromisoformat(due)
            except Exception:
                continue
            model = data.get("model", "")
            for fc in data.get("forecasts", []):
                if not fc.get("resolved") or fc.get("imputed"):
                    continue
                if fc.get("resolved_to") not in (0.0, 1.0) or fc.get("forecast") is None:
                    continue
                k = (fc["source"], str(fc["id"]), fc["resolution_date"], due)
                try:
                    hd = (datetime.fromisoformat(fc["resolution_date"]) - due_dt).days
                except Exception:
                    hd = None
                outcome[k] = fc["resolved_to"]
                meta[k] = (fc["source"], hbin(hd))
                if bname:
                    base_fc[bname][k] = fc["forecast"]
                else:
                    llm[model][k] = fc["forecast"]
    return dict(llm), dict(base_fc), outcome, meta


def climatology(outcome, meta):
    """Leave-one-out base rate of outcome=1 within each (source, horizon-bin)."""
    by_cell = defaultdict(list)
    for k, o in outcome.items():
        by_cell[meta[k]].append(o)
    sums = {c: (sum(v), len(v)) for c, v in by_cell.items()}
    return {k: ((sums[meta[k]][0] - o) / (sums[meta[k]][1] - 1) if sums[meta[k]][1] > 1
                else sums[meta[k]][0] / sums[meta[k]][1]) for k, o in outcome.items()}


def score_on(keys, fc_map, outcome):
    keys = [k for k in keys if k in fc_map]
    return brier([fc_map[k] for k in keys], [outcome[k] for k in keys]), len(keys)


def main():
    llm, base_fc, outcome, meta = load(BASE)
    climo = climatology(outcome, meta)
    crowd_map = {}  # mean LLM forecast per question (context, not a "simple" baseline)
    perq = defaultdict(list)
    for m, d in llm.items():
        for k, v in d.items():
            perq[k].append(v)
    crowd_map = {k: float(np.mean(v)) for k, v in perq.items()}
    print(f"loaded {len(llm)} LLMs, {len(outcome)} resolved binary questions")
    print(f"baselines: {list(base_fc)} | climatology + crowd computed\n")

    forecasters = {"climatology": climo, "naive": base_fc.get("naive", {}),
                   "imputed": base_fc.get("imputed", {}), "always0.5": base_fc.get("always0.5", {}),
                   "crowd(meanLLM)": crowd_map}

    for focus in FOCUS:
        if focus not in llm:
            print(f"[skip] {focus} not found"); continue
        fmap = llm[focus]
        kall = set(fmap)
        kds = {k for k in kall if meta[k][0] in DATASET_SRC}
        kmk = {k for k in kall if meta[k][0] in MARKET_SRC}
        print(f"\n{'='*78}\n{focus}\n{'='*78}")
        for label, ks in [("ALL", kall), ("DATASET srcs", kds), ("MARKET srcs", kmk)]:
            if not ks:
                continue
            b_llm, n = score_on(ks, fmap, outcome)
            line = f"  {label:<14} n={n:<5} LLM Brier={b_llm:.4f}  |  "
            comps = []
            for name, fmap2 in forecasters.items():
                b, _ = score_on(ks, fmap2, outcome)
                flag = "↑LLM" if b_llm < b else "↓base"
                comps.append(f"{name}={b:.4f}({flag})")
            print(line + "  ".join(comps))

    # yfinance by horizon — the volatile/anti-g source, our treasury analog
    print(f"\n{'='*78}\nyfinance by horizon: Llama-3.3-70B vs climatology vs naive\n{'='*78}")
    focus = FOCUS[0]
    if focus in llm:
        fmap = llm[focus]
        for _, _, lab in HBINS:
            ks = {k for k in fmap if meta[k][0] == "yfinance" and meta[k][1] == lab}
            if len(ks) < 15:
                continue
            bl, n = score_on(ks, fmap, outcome)
            bc, _ = score_on(ks, climo, outcome)
            bn, _ = score_on(ks, base_fc.get("naive", {}), outcome)
            br = np.mean([outcome[k] for k in ks])
            print(f"  {lab:<6} n={n:<4} base_rate(up)={br:.2f} | LLM={bl:.4f}  climo={bc:.4f}  "
                  f"naive={bn:.4f}  → {'LLM beats climo' if bl<bc else 'climo beats LLM'}")


if __name__ == "__main__":
    main()
