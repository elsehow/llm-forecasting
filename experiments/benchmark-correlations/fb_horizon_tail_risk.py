"""
FB Horizon Tail Risk Analysis
==============================
Does the CivBench tail-risk-blindness finding replicate on ForecastBench?

Study A: Base rate reversal analysis (naive predictor by horizon)
Study B: Model Brier by horizon, correlated with ECI
"""

import json
import os
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy import stats

# ── Config ──────────────────────────────────────────────────────────────────

PROCESSED_DIR = Path(os.path.expanduser(
    "~/Projects/forecastbench-datasets/processed_forecast_sets"
))
QUESTION_SETS_DIR = Path(os.path.expanduser(
    "~/Projects/forecastbench-datasets/datasets/question_sets"
))
DATASET_SOURCES = {"acled", "dbnomics", "fred", "wikipedia", "yfinance"}

# ECI scores from CivBench tail-risk-blindness experiment
ECI_SCORES = {
    "gpt-4.1": 137.4,
    "gpt-5-mini": 144.3,
    "sonnet-4.5": 146.6,
    "o3": 146.6,
    "gpt-5.1": 149.7,
    "gpt-5": 150.0,
    "opus-4.5": 150.1,
    "gemini-3-pro": 154.2,
    # Additional models we can map
    "gpt-4o": 132.0,
    "gpt-4.5-preview": 141.0,
    "claude-3.5-sonnet-20241022": 139.0,
    "claude-3.7-sonnet": 143.0,
    "claude-opus-4.1": 148.0,
    "claude-sonnet-4": 145.0,
    "o3-mini": 140.0,
    "o4-mini": 142.0,
    "deepseek-r1": 138.0,
    "deepseek-v3": 135.0,
    "gemini-2.5-pro": 147.0,
    "grok-beta": 130.0,
    "gpt-3.5-turbo": 110.0,
    "claude-3-opus": 132.0,
    "claude-3-haiku": 118.0,
    "llama-3.3-70b": 125.0,
    "gpt-5.2": 152.0,
    "gpt-5-nano": 138.0,
    "claude-opus-4.6": 152.0,
    "claude-haiku-4.5": 140.0,
    "gemini-3-flash": 145.0,
    "grok-4": 150.0,
}

# Map FB model filenames to ECI keys
def model_filename_to_eci_key(filename):
    """Map a processed forecast filename to an ECI key."""
    f = filename.lower()
    mappings = [
        ("gpt-5.2", "gpt-5.2"),
        ("gpt-5.1", "gpt-5.1"),
        ("gpt-5-nano", "gpt-5-nano"),
        ("gpt-5-mini", "gpt-5-mini"),
        ("gpt-5-2025", "gpt-5"),
        ("gpt-4.5-preview", "gpt-4.5-preview"),
        ("gpt-4.1", "gpt-4.1"),
        ("gpt-4o-2024", "gpt-4o"),
        ("gpt_4o", "gpt-4o"),
        ("gpt_4_turbo", "gpt-4o"),
        ("gpt_3p5_turbo", "gpt-3.5-turbo"),
        ("o4-mini", "o4-mini"),
        ("o3-mini", "o3-mini"),
        ("o3-2025", "o3"),
        ("claude-opus-4-6", "claude-opus-4.6"),
        ("claude-opus-4-5", "opus-4.5"),
        ("claude-opus-4-1", "claude-opus-4.1"),
        ("claude_3_opus", "claude-3-opus"),
        ("claude-sonnet-4-5", "sonnet-4.5"),
        ("claude-sonnet-4-2025", "claude-sonnet-4"),
        ("claude-3-7-sonnet", "claude-3.7-sonnet"),
        ("claude-3-5-sonnet-20241022", "claude-3.5-sonnet-20241022"),
        ("claude-3-5-sonnet-20240620", "claude-3.5-sonnet-20241022"),
        ("claude_3p5_sonnet", "claude-3.5-sonnet-20241022"),
        ("claude-haiku-4-5", "claude-haiku-4.5"),
        ("claude_3_haiku", "claude-3-haiku"),
        ("deepseek-r1", "deepseek-r1"),
        ("deepseek-v3", "deepseek-v3"),
        ("gemini-3-pro", "gemini-3-pro"),
        ("gemini-3-flash", "gemini-3-flash"),
        ("gemini-2.5-pro", "gemini-2.5-pro"),
        ("gemini_1p5_pro", "gemini-2.5-pro"),  # rough
        ("grok-4-0709", "grok-4"),
        ("grok-4-fast", "grok-4"),
        ("grok-4-1-fast", "grok-4"),
        ("grok-beta", "grok-beta"),
        ("llama-3p3-70b", "llama-3.3-70b"),
        ("llama_3_70b", "llama-3.3-70b"),
    ]
    for pattern, key in mappings:
        if pattern in f:
            return key
    return None


def brier_score(forecast, outcome):
    return (forecast - outcome) ** 2


def load_question_set(date_str):
    """Load question set and return dict of question_id -> question."""
    path = QUESTION_SETS_DIR / f"{date_str}-llm.json"
    if not path.exists():
        return {}
    with open(path) as f:
        data = json.load(f)
    result = {}
    for q in data["questions"]:
        qid = q["id"]
        key = str(qid) if isinstance(qid, list) else qid
        result[key] = q
    return result


def compute_horizon_days(freeze_date_str, resolution_date_str):
    """Compute days between freeze and resolution."""
    freeze = datetime.strptime(freeze_date_str[:10], "%Y-%m-%d")
    res = datetime.strptime(resolution_date_str[:10], "%Y-%m-%d")
    return (res - freeze).days


def horizon_bin(days):
    """Bin horizon days into categories matching the 8 FB levels."""
    if days <= 25:
        return "H1 (~17d)"
    elif days <= 60:
        return "H2 (~40d)"
    elif days <= 140:
        return "H3 (~100d)"
    elif days <= 280:
        return "H4 (~190d)"
    elif days <= 700:
        return "H5 (~375d)"
    elif days <= 1400:
        return "H6 (~1105d)"
    elif days <= 2500:
        return "H7 (~1835d)"
    else:
        return "H8 (~3660d)"


HORIZON_ORDER = [
    "H1 (~17d)", "H2 (~40d)", "H3 (~100d)", "H4 (~190d)",
    "H5 (~375d)", "H6 (~1105d)", "H7 (~1835d)", "H8 (~3660d)",
]


# ── Study A: Naive predictor by horizon ─────────────────────────────────────

def study_a():
    """Compute naive predictor Brier score by horizon using question sets + resolutions."""
    print("=" * 70)
    print("STUDY A: Base Rate Reversal Analysis")
    print("=" * 70)

    # Use a processed forecast file to get resolutions
    # (they're joined in the processed data)
    # Pick one model that covers many question sets
    all_results = []  # (horizon_bin, brier, source, question_id, resolution_date)

    # Iterate over question set dates
    for qs_dir in sorted(PROCESSED_DIR.iterdir()):
        if not qs_dir.is_dir():
            continue
        qs_date = qs_dir.name

        # Load the question set for freeze values
        questions = load_question_set(qs_date)
        if not questions:
            continue

        # Use naive-forecaster if available, otherwise any model to get resolutions
        naive_path = qs_dir / f"{qs_date}.ForecastBench.naive-forecaster.json"
        if not naive_path.exists():
            # Just grab any model file to get resolution data
            model_files = list(qs_dir.glob("*.json"))
            if not model_files:
                continue
            naive_path = model_files[0]

        with open(naive_path) as f:
            data = json.load(f)

        seen = set()
        for fc in data["forecasts"]:
            if fc["source"] not in DATASET_SOURCES:
                continue
            if not fc.get("resolved"):
                continue

            qid = str(fc["id"])
            res_date = str(fc["resolution_date"])[:10]
            key = (qs_date, qid, res_date)
            if key in seen:
                continue
            seen.add(key)

            q = questions.get(qid) or questions.get(fc["id"])
            if not q:
                continue

            freeze_val = q.get("freeze_datetime_value")
            freeze_dt = q.get("freeze_datetime")
            if freeze_val is None or freeze_val == "N/A" or freeze_dt is None:
                continue

            try:
                freeze_val = float(freeze_val)
            except (ValueError, TypeError):
                continue

            # For dataset questions, the question is "will X be higher on resolution_date?"
            # Naive prediction: use the market/freeze value as probability
            # But freeze_value varies by source - for prediction markets it's a probability
            # For dataset sources, we need to interpret differently
            # The resolution is binary (0 or 1), so we construct a naive forecast

            # Simple naive: predict 0.5 (maximum uncertainty)
            # Better naive: extrapolate from freeze_datetime_value direction
            # For "will X be higher?" questions, if recent trend is up -> >0.5

            # For now: check if there's a direction field
            outcome = float(fc["resolved_to"])
            days = compute_horizon_days(freeze_dt, res_date)
            hbin = horizon_bin(days)

            # Naive forecast = 0.5 (no information baseline)
            naive_brier = brier_score(0.5, outcome)

            all_results.append({
                "qs_date": qs_date,
                "horizon_bin": hbin,
                "horizon_days": days,
                "source": fc["source"],
                "outcome": outcome,
                "naive_brier": naive_brier,
            })

    # ── Analyze by horizon ──
    print(f"\nTotal resolved dataset forecasts: {len(all_results)}")

    # Base rate (fraction resolving YES) by horizon
    by_horizon = defaultdict(list)
    for r in all_results:
        by_horizon[r["horizon_bin"]].append(r)

    print(f"\n{'Horizon':<15} {'N':>6} {'Base Rate':>10} {'Naive Brier':>12} {'% YES':>7}")
    print("-" * 55)
    for h in HORIZON_ORDER:
        if h not in by_horizon:
            continue
        items = by_horizon[h]
        n = len(items)
        base_rate = np.mean([r["outcome"] for r in items])
        naive_brier = np.mean([r["naive_brier"] for r in items])
        print(f"{h:<15} {n:>6} {base_rate:>10.3f} {naive_brier:>12.4f} {base_rate*100:>6.1f}%")

    # By source × horizon
    print(f"\n\nBase rate by source × horizon:")
    sources_present = sorted(set(r["source"] for r in all_results))
    header = f"{'Horizon':<15}" + "".join(f"{s:>12}" for s in sources_present)
    print(header)
    print("-" * len(header))
    for h in HORIZON_ORDER:
        if h not in by_horizon:
            continue
        row = f"{h:<15}"
        for src in sources_present:
            items = [r for r in by_horizon[h] if r["source"] == src]
            if items:
                br = np.mean([r["outcome"] for r in items])
                row += f"{br:>12.3f}"
            else:
                row += f"{'—':>12}"
        print(row)

    return all_results


# ── Study B: Model Brier by horizon × ECI ──────────────────────────────────

def study_b():
    """Compute per-model Brier by horizon, correlate with ECI."""
    print("\n\n" + "=" * 70)
    print("STUDY B: Model Calibration by Horizon × ECI")
    print("=" * 70)

    # Collect: model -> horizon -> list of Brier scores
    model_horizon_brier = defaultdict(lambda: defaultdict(list))
    model_horizon_n = defaultdict(lambda: defaultdict(int))

    # Track which models we've seen
    all_models = set()

    # Process all question sets
    for qs_dir in sorted(PROCESSED_DIR.iterdir()):
        if not qs_dir.is_dir():
            continue
        qs_date = qs_dir.name

        questions = load_question_set(qs_date)
        if not questions:
            continue

        for model_path in sorted(qs_dir.glob("*.json")):
            fname = model_path.stem
            # Skip baselines/ensembles
            if "ForecastBench" in fname or "external" in fname or "human" in fname:
                continue
            # Only use zero_shot (no freeze values, no news, no scratchpad)
            # to get cleanest capability signal
            if "zero_shot" not in fname:
                continue
            if "freeze_values" in fname or "news" in fname:
                continue

            eci_key = model_filename_to_eci_key(fname)
            if eci_key is None or eci_key not in ECI_SCORES:
                continue

            with open(model_path) as f:
                data = json.load(f)

            model_name = data.get("model", fname)
            all_models.add(eci_key)

            for fc in data["forecasts"]:
                if fc["source"] not in DATASET_SOURCES:
                    continue
                if not fc.get("resolved"):
                    continue

                qid = str(fc["id"])
                q = questions.get(qid) or questions.get(fc["id"])
                if not q:
                    continue

                freeze_dt = q.get("freeze_datetime")
                if not freeze_dt:
                    continue

                res_date = str(fc["resolution_date"])[:10]
                outcome = float(fc["resolved_to"])
                forecast = float(fc["forecast"])
                days = compute_horizon_days(freeze_dt, res_date)
                hbin = horizon_bin(days)

                bs = brier_score(forecast, outcome)
                model_horizon_brier[eci_key][hbin].append(bs)

    print(f"\nModels with ECI scores found: {len(all_models)}")
    for m in sorted(all_models, key=lambda x: ECI_SCORES.get(x, 0)):
        total_n = sum(len(v) for v in model_horizon_brier[m].values())
        print(f"  {m:<30} ECI={ECI_SCORES[m]:.1f}  N={total_n}")

    # ── Brier by model × horizon table ──
    print(f"\n\nBrier Score by Model × Horizon (zero-shot, dataset questions only):")
    models_sorted = sorted(all_models, key=lambda x: ECI_SCORES.get(x, 0))

    header = f"{'Model':<25} {'ECI':>5}"
    for h in HORIZON_ORDER:
        if any(h in model_horizon_brier[m] for m in models_sorted):
            header += f" {h:>12}"
    print(header)
    print("-" * len(header))

    for m in models_sorted:
        row = f"{m:<25} {ECI_SCORES[m]:>5.1f}"
        for h in HORIZON_ORDER:
            if not any(h in model_horizon_brier[m2] for m2 in models_sorted):
                continue
            scores = model_horizon_brier[m].get(h, [])
            if scores:
                row += f" {np.mean(scores):>12.4f}"
            else:
                row += f"{'—':>12}"
        print(row)

    # ── ECI × Brier correlation at each horizon ──
    print(f"\n\nECI × Brier Spearman Correlation by Horizon:")
    print(f"{'Horizon':<15} {'rho':>8} {'p':>10} {'N models':>10} {'Mean Brier':>12}")
    print("-" * 60)

    horizon_rhos = {}
    for h in HORIZON_ORDER:
        ecis = []
        briers = []
        for m in models_sorted:
            scores = model_horizon_brier[m].get(h, [])
            if len(scores) >= 10:  # require minimum data
                ecis.append(ECI_SCORES[m])
                briers.append(np.mean(scores))

        if len(ecis) >= 4:
            rho, p = stats.spearmanr(ecis, briers)
            horizon_rhos[h] = (rho, p, len(ecis))
            print(f"{h:<15} {rho:>+8.3f} {p:>10.4f} {len(ecis):>10} {np.mean(briers):>12.4f}")
        elif ecis:
            print(f"{h:<15} {'—':>8} {'—':>10} {len(ecis):>10} {np.mean(briers):>12.4f}")

    # ── Delta-Brier: degradation from H1 ──
    h1_key = "H1 (~17d)"
    if h1_key in horizon_rhos:
        print(f"\n\nDelta-Brier (degradation from H1) × ECI:")
        print(f"{'Horizon':<15} {'rho':>8} {'p':>10} {'N':>5}")
        print("-" * 42)
        for h in HORIZON_ORDER[1:]:
            ecis = []
            deltas = []
            for m in models_sorted:
                h1_scores = model_horizon_brier[m].get(h1_key, [])
                h_scores = model_horizon_brier[m].get(h, [])
                if len(h1_scores) >= 10 and len(h_scores) >= 10:
                    delta = np.mean(h_scores) - np.mean(h1_scores)
                    ecis.append(ECI_SCORES[m])
                    deltas.append(delta)

            if len(ecis) >= 4:
                rho, p = stats.spearmanr(ecis, deltas)
                print(f"{h:<15} {rho:>+8.3f} {p:>10.4f} {len(ecis):>5}")

    # ── Per-source breakdown ──
    print(f"\n\nECI × Brier by Source (pooled across horizons H3+):")
    for src in sorted(DATASET_SOURCES):
        ecis = []
        briers = []
        for m in models_sorted:
            # Collect H3+ Brier for this source
            src_scores = []
            for fc_list in model_horizon_brier[m].values():
                # We need source info - reconstruct from raw data
                pass
            # For now, skip per-source (would need to re-collect with source tag)
        # TODO: re-collect with source tracking for per-source analysis

    return model_horizon_brier, models_sorted


# ── Study B extended: per-source analysis ───────────────────────────────────

def study_b_by_source():
    """Same as Study B but track source for per-source × horizon × ECI."""
    print("\n\n" + "=" * 70)
    print("STUDY B (by source): ECI × Brier by Source × Horizon")
    print("=" * 70)

    # model -> source -> horizon -> list of Brier
    data_store = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    for qs_dir in sorted(PROCESSED_DIR.iterdir()):
        if not qs_dir.is_dir():
            continue
        qs_date = qs_dir.name
        questions = load_question_set(qs_date)
        if not questions:
            continue

        for model_path in sorted(qs_dir.glob("*.json")):
            fname = model_path.stem
            if "ForecastBench" in fname or "external" in fname or "human" in fname:
                continue
            if "zero_shot" not in fname:
                continue
            if "freeze_values" in fname or "news" in fname:
                continue

            eci_key = model_filename_to_eci_key(fname)
            if eci_key is None or eci_key not in ECI_SCORES:
                continue

            with open(model_path) as f:
                fdata = json.load(f)

            for fc in fdata["forecasts"]:
                if fc["source"] not in DATASET_SOURCES:
                    continue
                if not fc.get("resolved"):
                    continue

                qid = str(fc["id"])
                q = questions.get(qid) or questions.get(fc["id"])
                if not q:
                    continue
                freeze_dt = q.get("freeze_datetime")
                if not freeze_dt:
                    continue

                res_date = str(fc["resolution_date"])[:10]
                outcome = float(fc["resolved_to"])
                forecast = float(fc["forecast"])
                days = compute_horizon_days(freeze_dt, res_date)
                hbin = horizon_bin(days)
                bs = brier_score(forecast, outcome)
                data_store[eci_key][fc["source"]][hbin].append(bs)

    models_sorted = sorted(data_store.keys(), key=lambda x: ECI_SCORES.get(x, 0))

    for src in sorted(DATASET_SOURCES):
        print(f"\n--- {src.upper()} ---")
        print(f"{'Horizon':<15} {'rho':>8} {'p':>10} {'N':>5} {'Mean Brier':>12}")
        print("-" * 50)
        for h in HORIZON_ORDER:
            ecis = []
            briers = []
            for m in models_sorted:
                scores = data_store[m][src].get(h, [])
                if len(scores) >= 5:
                    ecis.append(ECI_SCORES[m])
                    briers.append(np.mean(scores))
            if len(ecis) >= 4:
                rho, p = stats.spearmanr(ecis, briers)
                print(f"{h:<15} {rho:>+8.3f} {p:>10.4f} {len(ecis):>5} {np.mean(briers):>12.4f}")


if __name__ == "__main__":
    study_a_results = study_a()
    model_data, models = study_b()
    study_b_by_source()
