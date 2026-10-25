"""
Reanalyze ForecastBench forecasts for the tailriskbench paper.

Test: does the capability-overconfidence pattern appear in ForecastBench binary forecasts?
- Binary Brier should be pro-g (more capable = lower Brier)
- But calibration (reliability) may be flat or anti-g
- Overconfidence should be worse on volatile sources (yfinance) and longer horizons
"""

import json
import os
import numpy as np
from collections import defaultdict
from datetime import datetime
from scipy import stats

# ── ECI scores keyed by normalized ForecastBench model names ──
# Source: civbench/analyze_eci_correlation.py + Epoch AI
ECI_BY_FB_NAME = {
    # Anthropic
    "claude-2.1": 112.0,
    "claude-3-haiku-20240307": 118.0,
    "claude-3-opus-20240229": 130.0,
    "claude-3-5-sonnet-20240620": 133.0,
    "claude-3-5-sonnet-20241022": 135.0,
    "claude-3-7-sonnet-20250219": 140.0,
    "claude-haiku-4-5-20251001": 140.0,
    "claude-sonnet-4-20250514": 145.0,
    "claude-opus-4-20250514": 147.0,
    "claude-sonnet-4-5-20250929": 146.6,
    "claude-opus-4-1-20250805": 149.0,
    "claude-opus-4-5-20251101": 150.1,
    "claude-opus-4-6": 152.0,
    # OpenAI
    "gpt-3.5-turbo-0125": 110.0,
    "gpt-4-0613": 120.0,
    "gpt-4-turbo-2024-04-09": 128.0,
    "gpt-4o": 132.0,
    "gpt-4o-2024-05-13": 132.0,
    "gpt-4.1-2025-04-14": 137.4,
    "gpt-4.5-preview-2025-02-27": 140.0,
    "gpt-5-nano-2025-08-07": 138.0,
    "gpt-5-mini-2025-08-07": 144.3,
    "gpt-5-2025-08-07": 150.0,
    "gpt-5.1-2025-11-13": 149.7,
    "gpt-5.2-2025-12-11": 151.0,
    "o3-mini-2025-01-31": 140.0,
    "o3-2025-04-16": 146.6,
    "o4-mini-2025-04-16": 142.0,
    # Google
    "gemini-1.5-flash": 122.0,
    "gemini-1.5-pro": 128.0,
    "gemini-2.0-flash-lite-001": 125.0,
    "gemini-2.5-flash": 145.0,
    "gemini-2.5-flash-preview-04-17": 145.0,
    "gemini-2.5-pro": 147.0,
    "gemini-2.5-pro-exp-03-25": 147.0,
    "gemini-2.5-pro-preview-03-25": 147.0,
    "gemini-3-flash-preview": 148.0,
    "gemini-3-pro-preview": 154.2,
    # xAI
    "grok-beta": 125.0,
    "grok-4-0709": 150.0,
    "grok-4-fast-reasoning": 150.0,
    "grok-4-fast-non-reasoning": 145.0,
    "grok-4-1-fast-reasoning": 151.0,
    "grok-4-1-fast-non-reasoning": 146.0,
    # DeepSeek
    "deepseek-r1": 138.0,
    "deepseek-v3": 135.0,
    "deepseek-v3.1": 137.0,
    # Meta
    "llama-2-70b-chat-hf": 105.0,
    "llama-3-70b-chat-hf": 118.0,
    "llama-3-8b-chat-hf": 108.0,
    "llama-3.2-3b-instruct-turbo": 105.0,
    "llama-3.3-70b-instruct-turbo": 125.0,
    "llama-4-maverick-17b-128e-instruct-fp8": 132.0,
    "llama-4-scout-17b-16e-instruct": 128.0,
    "meta-llama-3.1-405b-instruct-turbo": 130.0,
    # Mistral
    "mistral-large-2407": 128.0,
    "mistral-large-2411": 130.0,
    "mistral-large-latest": 130.0,
    "magistral-medium-2506": 132.0,
    "mixtral-8x22b-instruct-v0.1": 118.0,
    "mixtral-8x7b-instruct-v0.1": 110.0,
    # Others
    "kimi-k2-instruct": 140.0,
    "kimi-k2-instruct-0905": 140.0,
    "kimi-k2-thinking": 142.0,
    "glm-4.5-air-fp8": 125.0,
    "glm-4.6": 128.0,
    "glm-4.7": 130.0,
    "qwq-32b-preview": 130.0,
    "qwen1.5-110b-chat": 118.0,
    "qwen2.5-72b-instruct-turbo": 128.0,
    "qwen3-235b-a22b-fp8-tput": 138.0,
    "qwen3-235b-a22b-thinking-2507": 140.0,
}


def match_eci(model_name):
    """Match a ForecastBench model name to an ECI score (case-insensitive)."""
    # Strip condition suffix like " (zero shot)"
    name = model_name.split(" (")[0].strip().lower()
    for key, eci in ECI_BY_FB_NAME.items():
        if key == name:
            return eci
    return None


def load_all_forecasts(base_dir, condition="zero_shot"):
    """Load all resolved forecasts matching the given condition."""
    # model_id -> list of {forecast, resolved_to, source, horizon_days, ...}
    model_data = defaultdict(list)

    for date_dir in sorted(os.listdir(base_dir)):
        dp = os.path.join(base_dir, date_dir)
        if not os.path.isdir(dp):
            continue

        for fname in sorted(os.listdir(dp)):
            if not fname.endswith(".json"):
                continue
            # Skip baselines and external
            if any(
                x in fname
                for x in [
                    "Always-",
                    "Random",
                    "Imputed",
                    "Naive",
                    "Baselines",
                    "external.",
                ]
            ):
                continue
            # Match condition
            if condition and f"_{condition}.json" not in fname:
                continue

            fp = os.path.join(dp, fname)
            with open(fp) as fh:
                data = json.load(fh)

            model_name = data.get("model", "")
            due_date = datetime.fromisoformat(data["forecast_due_date"])

            for fc in data["forecasts"]:
                if not fc.get("resolved"):
                    continue
                if fc.get("imputed"):
                    continue

                resolved_to = fc["resolved_to"]
                # Only binary resolutions
                if resolved_to not in (0.0, 1.0):
                    continue

                forecast = fc["forecast"]
                if forecast is None:
                    continue

                res_date = datetime.fromisoformat(fc["resolution_date"])
                horizon_days = (res_date - due_date).days

                model_data[model_name].append(
                    {
                        "forecast": forecast,
                        "outcome": resolved_to,
                        "source": fc["source"],
                        "horizon_days": horizon_days,
                        "date": date_dir,
                        "id": fc["id"],
                    }
                )

    return dict(model_data)


def brier_score(forecasts, outcomes):
    return np.mean((np.array(forecasts) - np.array(outcomes)) ** 2)


def murphy_decomposition(forecasts, outcomes, n_bins=10):
    """Murphy decomposition: Brier = Reliability - Resolution + Uncertainty"""
    f = np.array(forecasts)
    o = np.array(outcomes)
    n = len(f)
    base_rate = np.mean(o)
    uncertainty = base_rate * (1 - base_rate)

    # Bin forecasts
    bins = np.linspace(0, 1, n_bins + 1)
    reliability = 0.0
    resolution = 0.0

    for i in range(n_bins):
        mask = (f >= bins[i]) & (f < bins[i + 1])
        if i == n_bins - 1:  # include 1.0 in last bin
            mask = (f >= bins[i]) & (f <= bins[i + 1])
        nk = np.sum(mask)
        if nk == 0:
            continue
        fk = np.mean(f[mask])
        ok = np.mean(o[mask])
        reliability += nk * (fk - ok) ** 2
        resolution += nk * (ok - base_rate) ** 2

    reliability /= n
    resolution /= n

    return {
        "brier": brier_score(forecasts, outcomes),
        "reliability": reliability,  # lower = better calibrated
        "resolution": resolution,  # higher = better discrimination
        "uncertainty": uncertainty,
    }


def ece(forecasts, outcomes, n_bins=10):
    """Expected calibration error."""
    f = np.array(forecasts)
    o = np.array(outcomes)
    n = len(f)
    bins = np.linspace(0, 1, n_bins + 1)
    total = 0.0
    for i in range(n_bins):
        mask = (f >= bins[i]) & (f < bins[i + 1])
        if i == n_bins - 1:
            mask = (f >= bins[i]) & (f <= bins[i + 1])
        nk = np.sum(mask)
        if nk == 0:
            continue
        fk = np.mean(f[mask])
        ok = np.mean(o[mask])
        total += nk * abs(fk - ok)
    return total / n


def forecast_extremity(forecasts):
    """Mean |forecast - 0.5|: how far from maximum uncertainty."""
    return np.mean(np.abs(np.array(forecasts) - 0.5))


# ── Main ──
if __name__ == "__main__":
    BASE = "/Users/elsehow/Projects/forecastbench-datasets/processed_forecast_sets"

    print("Loading forecasts (zero_shot condition)...")
    model_data = load_all_forecasts(BASE, condition="zero_shot")
    print(f"Loaded {len(model_data)} models\n")

    # ── Per-model metrics ──
    results = []
    for model_name, entries in model_data.items():
        eci = match_eci(model_name)
        if eci is None:
            continue

        forecasts = [e["forecast"] for e in entries]
        outcomes = [e["outcome"] for e in entries]

        if len(forecasts) < 100:
            continue

        murphy = murphy_decomposition(forecasts, outcomes)
        extremity = forecast_extremity(forecasts)
        cal_error = ece(forecasts, outcomes)

        # yfinance subset
        yf_entries = [e for e in entries if e["source"] == "yfinance"]
        yf_forecasts = [e["forecast"] for e in yf_entries]
        yf_outcomes = [e["outcome"] for e in yf_entries]

        # non-yfinance subset
        other_entries = [e for e in entries if e["source"] != "yfinance"]
        other_forecasts = [e["forecast"] for e in other_entries]
        other_outcomes = [e["outcome"] for e in other_entries]

        results.append(
            {
                "model": model_name,
                "eci": eci,
                "n": len(forecasts),
                "brier": murphy["brier"],
                "reliability": murphy["reliability"],
                "resolution": murphy["resolution"],
                "uncertainty": murphy["uncertainty"],
                "ece": cal_error,
                "extremity": extremity,
                "n_yf": len(yf_forecasts),
                "brier_yf": brier_score(yf_forecasts, yf_outcomes)
                if yf_forecasts
                else None,
                "ece_yf": ece(yf_forecasts, yf_outcomes) if yf_forecasts else None,
                "extremity_yf": forecast_extremity(yf_forecasts)
                if yf_forecasts
                else None,
                "n_other": len(other_forecasts),
                "brier_other": brier_score(other_forecasts, other_outcomes)
                if other_forecasts
                else None,
                "ece_other": ece(other_forecasts, other_outcomes)
                if other_forecasts
                else None,
                "extremity_other": forecast_extremity(other_forecasts)
                if other_forecasts
                else None,
            }
        )

    results.sort(key=lambda x: x["eci"])

    # ── Print results ──
    print(f"{'Model':<45s} {'ECI':>5s} {'N':>6s} {'Brier':>6s} {'Reliab':>7s} {'Resol':>6s} {'ECE':>6s} {'Extrem':>7s}")
    print("-" * 100)
    for r in results:
        print(
            f"{r['model'][:44]:<45s} {r['eci']:5.0f} {r['n']:6d} {r['brier']:.4f} {r['reliability']:.5f} {r['resolution']:.4f} {r['ece']:.4f} {r['extremity']:.4f}"
        )

    # ── ECI correlations ──
    ecis = [r["eci"] for r in results]
    n_models = len(ecis)
    print(f"\n{'='*80}")
    print(f"ECI CORRELATIONS (N={n_models} models)")
    print(f"{'='*80}")

    metrics = [
        ("Brier (lower=better)", "brier", False),
        ("Reliability (lower=better cal)", "reliability", False),
        ("Resolution (higher=better disc)", "resolution", True),
        ("ECE (lower=better cal)", "ece", False),
        ("Extremity (|f-0.5|)", "extremity", None),
        ("Brier (yfinance)", "brier_yf", False),
        ("ECE (yfinance)", "ece_yf", False),
        ("Extremity (yfinance)", "extremity_yf", None),
        ("Brier (other sources)", "brier_other", False),
        ("ECE (other sources)", "ece_other", False),
        ("Extremity (other sources)", "extremity_other", None),
    ]

    for label, key, _ in metrics:
        vals = [r[key] for r in results if r[key] is not None]
        ecis_filtered = [r["eci"] for r in results if r[key] is not None]
        if len(vals) < 5:
            continue
        rho, p = stats.spearmanr(ecis_filtered, vals)
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        print(f"  ECI × {label:<35s}: ρ={rho:+.3f}  p={p:.4f} {sig}")

    # ── Horizon analysis for yfinance ──
    print(f"\n{'='*80}")
    print("YFINANCE: CALIBRATION BY HORIZON")
    print(f"{'='*80}")

    horizon_bins = [(0, 10, "1wk"), (25, 35, "1mo"), (80, 100, "3mo"), (170, 190, "6mo"), (350, 380, "12mo")]

    for h_lo, h_hi, h_label in horizon_bins:
        h_results = []
        for model_name, entries in model_data.items():
            eci = match_eci(model_name)
            if eci is None:
                continue
            yf = [e for e in entries if e["source"] == "yfinance" and h_lo <= e["horizon_days"] <= h_hi]
            if len(yf) < 20:
                continue
            forecasts = [e["forecast"] for e in yf]
            outcomes = [e["outcome"] for e in yf]
            h_results.append(
                {
                    "model": model_name,
                    "eci": eci,
                    "n": len(yf),
                    "brier": brier_score(forecasts, outcomes),
                    "ece": ece(forecasts, outcomes, n_bins=5),
                    "extremity": forecast_extremity(forecasts),
                    "base_rate": np.mean(outcomes),
                    "mean_forecast": np.mean(forecasts),
                }
            )

        if len(h_results) < 5:
            print(f"\n  {h_label}: insufficient models ({len(h_results)})")
            continue

        ecis_h = [r["eci"] for r in h_results]
        briers_h = [r["brier"] for r in h_results]
        eces_h = [r["ece"] for r in h_results]
        exts_h = [r["extremity"] for r in h_results]

        rho_b, p_b = stats.spearmanr(ecis_h, briers_h)
        rho_e, p_e = stats.spearmanr(ecis_h, eces_h)
        rho_x, p_x = stats.spearmanr(ecis_h, exts_h)
        mean_br = np.mean([r["base_rate"] for r in h_results])
        mean_fc = np.mean([r["mean_forecast"] for r in h_results])

        sig_b = "***" if p_b < 0.001 else "**" if p_b < 0.01 else "*" if p_b < 0.05 else ""
        sig_e = "***" if p_e < 0.001 else "**" if p_e < 0.01 else "*" if p_e < 0.05 else ""
        sig_x = "***" if p_x < 0.001 else "**" if p_x < 0.01 else "*" if p_x < 0.05 else ""

        print(f"\n  {h_label} (N={len(h_results)} models, base_rate={mean_br:.3f}, mean_forecast={mean_fc:.3f}):")
        print(f"    ECI × Brier:     ρ={rho_b:+.3f}  p={p_b:.4f} {sig_b}")
        print(f"    ECI × ECE:       ρ={rho_e:+.3f}  p={p_e:.4f} {sig_e}")
        print(f"    ECI × Extremity: ρ={rho_x:+.3f}  p={p_x:.4f} {sig_x}")

    # ── Regime-conditional: stocks that went DOWN ──
    print(f"\n{'='*80}")
    print("REGIME-CONDITIONAL: OVERCONFIDENCE WHEN STOCKS FELL")
    print(f"{'='*80}")

    for h_lo, h_hi, h_label in horizon_bins:
        h_results = []
        for model_name, entries in model_data.items():
            eci = match_eci(model_name)
            if eci is None:
                continue
            # yfinance questions where stock went DOWN (resolved_to = 0)
            yf_down = [
                e for e in entries
                if e["source"] == "yfinance"
                and h_lo <= e["horizon_days"] <= h_hi
                and e["outcome"] == 0.0
            ]
            if len(yf_down) < 10:
                continue
            forecasts_down = [e["forecast"] for e in yf_down]
            # Mean forecast when stock actually fell — higher = more overconfident
            h_results.append(
                {
                    "model": model_name,
                    "eci": eci,
                    "n": len(yf_down),
                    "mean_forecast_when_fell": np.mean(forecasts_down),
                    "brier_when_fell": brier_score(forecasts_down, [0.0] * len(yf_down)),
                }
            )

        if len(h_results) < 5:
            print(f"\n  {h_label}: insufficient models ({len(h_results)})")
            continue

        ecis_h = [r["eci"] for r in h_results]
        overconf = [r["mean_forecast_when_fell"] for r in h_results]
        briers_down = [r["brier_when_fell"] for r in h_results]

        rho_o, p_o = stats.spearmanr(ecis_h, overconf)
        rho_bd, p_bd = stats.spearmanr(ecis_h, briers_down)
        mean_oc = np.mean(overconf)

        sig_o = "***" if p_o < 0.001 else "**" if p_o < 0.01 else "*" if p_o < 0.05 else ""
        sig_bd = "***" if p_bd < 0.001 else "**" if p_bd < 0.01 else "*" if p_bd < 0.05 else ""

        print(f"\n  {h_label} (N={len(h_results)} models, mean P(up) when stock fell={mean_oc:.3f}):")
        print(f"    ECI × P(up|fell):  ρ={rho_o:+.3f}  p={p_o:.4f} {sig_o}")
        print(f"    ECI × Brier|fell:  ρ={rho_bd:+.3f}  p={p_bd:.4f} {sig_bd}")

    # ── Source comparison ──
    print(f"\n{'='*80}")
    print("CALIBRATION BY SOURCE")
    print(f"{'='*80}")

    sources = ["yfinance", "fred", "dbnomics", "acled", "manifold", "metaculus", "polymarket", "wikipedia"]
    for src in sources:
        src_ecis = []
        src_briers = []
        src_eces = []
        src_exts = []
        n_total = 0

        for model_name, entries in model_data.items():
            eci = match_eci(model_name)
            if eci is None:
                continue
            src_entries = [e for e in entries if e["source"] == src]
            if len(src_entries) < 20:
                continue
            forecasts = [e["forecast"] for e in src_entries]
            outcomes = [e["outcome"] for e in src_entries]
            src_ecis.append(eci)
            src_briers.append(brier_score(forecasts, outcomes))
            src_eces.append(ece(forecasts, outcomes, n_bins=5))
            src_exts.append(forecast_extremity(forecasts))
            n_total += len(src_entries)

        if len(src_ecis) < 5:
            continue

        rho_b, p_b = stats.spearmanr(src_ecis, src_briers)
        rho_e, p_e = stats.spearmanr(src_ecis, src_eces)
        rho_x, p_x = stats.spearmanr(src_ecis, src_exts)

        sig_b = "***" if p_b < 0.001 else "**" if p_b < 0.01 else "*" if p_b < 0.05 else ""
        sig_e = "***" if p_e < 0.001 else "**" if p_e < 0.01 else "*" if p_e < 0.05 else ""

        mean_brier = np.mean(src_briers)
        mean_ece = np.mean(src_eces)
        mean_ext = np.mean(src_exts)

        print(f"\n  {src} (N={len(src_ecis)} models, {n_total} forecasts, mean Brier={mean_brier:.4f}, mean ECE={mean_ece:.4f}):")
        print(f"    ECI × Brier:     ρ={rho_b:+.3f}  p={p_b:.4f} {sig_b}")
        print(f"    ECI × ECE:       ρ={rho_e:+.3f}  p={p_e:.4f} {sig_e}")
        print(f"    ECI × Extremity: ρ={rho_x:+.3f}  p={p_x:.4f}")
