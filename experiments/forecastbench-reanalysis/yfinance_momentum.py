"""Does Llama-3.3-70B's medium-horizon yfinance edge survive a momentum baseline?

The base-rate-vs-LLM comparison showed the 70B beats a flat climatology at 3-6mo on
yfinance. But neither climatology nor FB's naive baseline uses the obvious signal:
MOMENTUM (a stock rising into the freeze tends to keep rising). If a one-line momentum
model matches the 70B, the "edge" is momentum it got for free — not skill worth
mech-interping. This is the gate (same discipline that killed the smoke + the prime).

Baselines (all leave-one-out, self-contained — built from the freeze prices the
questions resolve against; no external API):
  - climatology : P(up | horizon)            — the flat base rate
  - momentum    : P(up | sign(trailing ret), horizon)  — deployable momentum
  - cohort      : P(up | due-date window, horizon)      — DIAGNOSTIC ONLY (hindsight:
                  the market direction of the resolution window, which a frozen model
                  cannot legitimately know). Bounds the systematic vs idiosyncratic split.

    cd /Users/elsehow/Projects/llm-forecasting
    uv run python experiments/forecastbench-reanalysis/yfinance_momentum.py
"""
import glob
import json
import os
from collections import defaultdict
from datetime import datetime

import numpy as np

DATASETS = "/Users/elsehow/Projects/forecastbench-datasets"
FCAST = f"{DATASETS}/processed_forecast_sets"
FOCUS = "Llama-3.3-70B-Instruct-Turbo (zero shot)"
# tight bins, matching the earlier yfinance-by-horizon analysis
HBINS = [(0, 10, "1wk"), (25, 35, "1mo"), (80, 100, "3mo"), (170, 190, "6mo"), (350, 380, "12mo")]


def hbin(d):
    for lo, hi, lab in HBINS:
        if d is not None and lo <= d <= hi:
            return lab
    return None


# 1) per-ticker freeze-price series from question sets
series = defaultdict(dict)
for qf in sorted(glob.glob(f"{DATASETS}/datasets/question_sets/*.json")):
    d = json.load(open(qf))
    due = d.get("forecast_due_date")
    for q in d.get("questions", []):
        if q.get("source") != "yfinance":
            continue
        try:
            series[q["id"]][due] = float(q.get("freeze_datetime_value"))
        except (TypeError, ValueError):
            pass


def trailing_return(tkr, due, max_gap=75):
    s = series.get(tkr, {})
    if due not in s:
        return None
    prior = [d for d in s if d < due]
    if not prior:
        return None
    pd_ = max(prior)
    if (datetime.fromisoformat(due) - datetime.fromisoformat(pd_)).days > max_gap:
        return None
    if s[pd_] <= 0:
        return None
    return s[due] / s[pd_] - 1


def sign(x):
    return "?" if x is None else ("up" if x > 0.005 else "dn" if x < -0.005 else "flat")


# 2) load 70B yfinance forecasts, joined with trailing return
rows = []
for dd in sorted(os.listdir(FCAST)):
    dp = os.path.join(FCAST, dd)
    if not os.path.isdir(dp):
        continue
    for f in os.listdir(dp):
        if "_zero_shot.json" not in f or "with_freeze_values" in f:
            continue
        try:
            data = json.load(open(os.path.join(dp, f)))
        except Exception:
            continue
        if data.get("model") != FOCUS:
            continue
        due = data["forecast_due_date"]
        for fc in data.get("forecasts", []):
            if fc.get("source") != "yfinance" or not fc.get("resolved") or fc.get("imputed"):
                continue
            if fc.get("resolved_to") not in (0.0, 1.0) or fc.get("forecast") is None:
                continue
            if not isinstance(fc.get("id"), str):  # skip combination/pair questions (list id)
                continue
            try:
                hd = (datetime.fromisoformat(fc["resolution_date"]) - datetime.fromisoformat(due)).days
            except Exception:
                hd = None
            b = hbin(hd)
            if b is None:
                continue
            rows.append({"id": fc["id"], "due": due, "hbin": b, "outcome": fc["resolved_to"],
                         "llm": fc["forecast"], "tret": trailing_return(fc["id"], due)})

ntret = sum(1 for r in rows if r["tret"] is not None)
print(f"70B yfinance rows (in tight horizon bins): {len(rows)} | with trailing return: {ntret}")


def make_loo(keyfn, rset_rows):
    grp = defaultdict(lambda: [0.0, 0])
    for r in rset_rows:
        g = grp[keyfn(r)]
        g[0] += r["outcome"]; g[1] += 1
    def pred(r):
        s, n = grp[keyfn(r)]
        return (s - r["outcome"]) / (n - 1) if n > 1 else 0.5
    return pred


def brier(rs, predfn):
    return float(np.mean([(predfn(r) - r["outcome"]) ** 2 for r in rs])) if rs else float("nan")


climo = make_loo(lambda r: r["hbin"], rows)
cohort = make_loo(lambda r: (r["due"], r["hbin"]), rows)

print(f"\n{'horizon':<7} {'n':>5} {'base(up)':>8} | {'70B':>6} {'climo':>6} {'cohort*':>7} || "
      f"{'momentum subset':<28}")
print("-" * 92)
for _, _, lab in HBINS:
    rs = [r for r in rows if r["hbin"] == lab]
    if len(rs) < 15:
        continue
    br = np.mean([r["outcome"] for r in rs])
    b_llm, b_cl, b_co = brier(rs, lambda r: r["llm"]), brier(rs, climo), brier(rs, cohort)
    # momentum on its own subset (rows with a trailing return), 70B re-scored on same subset
    rsm = [r for r in rs if r["tret"] is not None]
    mom = make_loo(lambda r: (sign(r["tret"]), r["hbin"]), rsm)
    b_llm_m, b_mom = brier(rsm, lambda r: r["llm"]), brier(rsm, mom)
    verdict = "70B>mom" if b_llm_m < b_mom else "MOM>=70B"
    print(f"{lab:<7} {len(rs):>5} {br:>8.2f} | {b_llm:>6.3f} {b_cl:>6.3f} {b_co:>7.3f} || "
          f"n={len(rsm):<4} 70B={b_llm_m:.3f} momentum={b_mom:.3f}  [{verdict}]")

print("\n* cohort = hindsight market-direction of the resolution window (diagnostic, not deployable).")
print("Gate: if momentum matches/beats the 70B at 3mo/6mo, the 'edge' is momentum, not skill.")
