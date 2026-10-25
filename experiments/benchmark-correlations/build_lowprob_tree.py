#!/usr/bin/env python3
"""Build an eval tree for the PRE-REGISTERED low-probability gate test.

PRE-REGISTERED GATES (fixed before any model responses are seen), applied per
question class across the 13 frontier models:
  G1 reliability:        split-half (even/odd instances) Spearman of the model
                         ranking, Spearman-Brown corrected, >= 0.5
  G2 beats-trivial:      best-model Brier <= 0.9 x flat-base-rate Brier on the
                         same instances
  G3 style-independence: |corr(mean |p-0.5|, class skill)| <= 0.35
A class must pass all three to enter any aggregate/proxy score. Transfer to FB
(conditional on ECI) is then tested only on the surviving composite.

Design: sample N_GAMES games (seed 13) from the mined low-prob corpus
(35 classes, 1-9% base-rate band, snapshot turn 40); take ALL selected-class
instances in those games so each game contributes one batched prompt; build
world reports at turn 40 directly from data/games (recordings are gone).
"""
import json
import numpy as np
from pathlib import Path

N_GAMES = 45
SEED = 13
SNAP = 40
ROOT = Path("/Users/elsehow/Projects/forecastbench-sim")
TREE = Path("/private/tmp/claude-501/-Users-elsehow-Projects-s-tier/f2b46dcd-fe73-45c4-8c11-533351736920/scratchpad/lowprob-tree")

METRICS = [  # (time_series key, label)
    ("scores", "SCORES"), ("treasury", "TREASURY"), ("population", "POPULATION"),
    ("techs_known", "TECHS KNOWN"), ("wonders_count", "WONDERS COUNT"),
    ("cities_count", "CITIES COUNT"), ("territory_size", "TERRITORY SIZE"),
    ("science", "SCIENCE"), ("trade_production", "TRADE"),
    ("food_production", "FOOD"), ("shield_production", "SHIELDS"),
    ("units_count", "UNITS"), ("military_units_count", "MILITARY UNITS"),
]
EVENT_TYPES = {"tech_discovered", "wonder_completed", "government_change",
               "city_founded", "city_conquered", "city_destroyed"}

def build_report(game_id: str, gd: dict, turn: int) -> str:
    civs = gd["civilizations"]
    ts = gd["time_series"]
    sampled = [1] + list(range(5, turn + 1, 5))
    L = [f"WORLD REPORT TXT v1 (data-only rebuild; no recordings)",
         f"Game ID: {game_id}", f"Snapshot turn: {turn}",
         f"Turns analyzed: 1..{turn}", "", "CIVILIZATIONS", "ID | Name"]
    for cid in sorted(civs, key=int):
        L.append(f"{cid}  | {civs[cid].get('name','?')}")
    L.append("")
    # time_series shape: [metric][turn(str)][player_id(str)] -> value
    def val(series, cid, t):
        return series.get(str(t), {}).get(str(cid), "NA")
    for key, label in METRICS:
        if key not in ts:
            continue
        active = sorted({c for t in sampled for c in ts[key].get(str(t), {})}, key=int)
        if not active:
            continue
        L.append(f"{label} (sampled every 5 turns)")
        L.append("turns: " + ", ".join(str(t) for t in sampled))
        for cid in active:
            row = ", ".join(str(val(ts[key], cid, t)) for t in sampled)
            L.append(f"{cid} {civs.get(cid, {}).get('name', '?')}: {row}")
        L.append("")
    L.append(f"EVENTS (through turn {turn})")
    for e in gd.get("events", []):
        if e.get("turn", 1e9) <= turn and e.get("type") in EVENT_TYPES:
            L.append(f"turn {e['turn']}: {e.get('description','')}")
    return "\n".join(L)

def main():
    d = json.loads((ROOT / "data/lowprob/lowprob_questions.json").read_text())
    qs = d["questions"]
    games = sorted({q["game_id"] for q in qs})
    rng = np.random.default_rng(SEED)
    chosen = set(rng.choice(games, size=N_GAMES, replace=False).tolist())
    picked = [q for q in qs if q["game_id"] in chosen]
    print(f"{len(chosen)} games, {len(picked)} questions, "
          f"{len({(q['template_id'], q.get('target',''), q['horizon']) for q in picked})} classes present")

    TREE.mkdir(parents=True, exist_ok=True)
    out_qs = []
    for q in picked:
        out_qs.append({
            "game_id": q["game_id"],
            "question_id": f"{q['game_id']}_{q['question_id']}",
            "template_id": q["template_id"],
            "question_type": "binary",
            "question_text": q["question_text"],
            "horizon": q["horizon"],
            "resolution": {"answer": bool(q["ground_truth"])},
            "parameters": {"game_id": q["game_id"], "snapshot_turn": SNAP,
                           "target": q.get("target", ""),
                           "class_base_rate": q["class_base_rate"],
                           "resolution_turn": q["resolution_turn"]},
        })
    (TREE / "questions_all.json").write_text(json.dumps({
        "metadata": {"purpose": "lowprob gate test", "seed": SEED,
                     "n_games": N_GAMES, "snapshot_turn": SNAP,
                     "num_questions": len(out_qs)},
        "questions": out_qs}, indent=1))
    for gid in sorted(chosen):
        gd = json.loads((ROOT / f"data/games/{gid}_data.json").read_text())
        rep_dir = TREE / gid / "world_report"
        rep_dir.mkdir(parents=True, exist_ok=True)
        (rep_dir / f"turn_{SNAP:03d}_report.txt").write_text(build_report(gid, gd, SNAP))
    print(f"tree -> {TREE}")

if __name__ == "__main__":
    main()
