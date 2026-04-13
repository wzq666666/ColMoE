#!/usr/bin/env python3
"""
Aggregate per-layer metrics from a jsonl file of MoE stats.

Usage:
  python tool/analyze_moe_layers.py /path/to/gsm8k.moe_layers.jsonl

The input file is expected to contain one JSON object per line with keys:
  - layer_hs_pct_avg: {layer_idx_str: float}
  - layer_conc_ratio_avg: {layer_idx_str: float}
  - random_expected_pct: float (optional)

Outputs a JSON summary to stdout with:
  - num_records: lines successfully parsed
  - random_expected_pct_mean: average of random_expected_pct (if present)
  - layer_hs_pct_mean: {layer_idx: mean across records}
  - layer_conc_ratio_mean: {layer_idx: mean across records}
"""

import argparse
import json
from collections import defaultdict
from typing import Dict, Iterable, Tuple


def _accumulate(
    lines: Iterable[str],
) -> Tuple[Dict[int, float], Dict[int, int], Dict[int, float], Dict[int, int], float, int]:
    """Parse lines and accumulate sums and counts per layer."""
    hs_sum: Dict[int, float] = defaultdict(float)
    hs_cnt: Dict[int, int] = defaultdict(int)
    conc_sum: Dict[int, float] = defaultdict(float)
    conc_cnt: Dict[int, int] = defaultdict(int)
    random_sum = 0.0
    random_cnt = 0

    for raw in lines:
        if not raw.strip():
            continue
        try:
            obj = json.loads(raw)
        except Exception:
            continue

        hs = obj.get("layer_hs_pct_avg") or {}
        conc = obj.get("layer_conc_ratio_avg") or {}
        rnd = obj.get("random_expected_pct")

        for k, v in hs.items():
            try:
                idx = int(k)
                hv = float(v)
            except (ValueError, TypeError):
                continue
            hs_sum[idx] += hv
            hs_cnt[idx] += 1

        for k, v in conc.items():
            try:
                idx = int(k)
                cv = float(v)
            except (ValueError, TypeError):
                continue
            conc_sum[idx] += cv
            conc_cnt[idx] += 1

        if rnd is not None:
            try:
                random_sum += float(rnd)
                random_cnt += 1
            except (ValueError, TypeError):
                pass

    return hs_sum, hs_cnt, conc_sum, conc_cnt, random_sum, random_cnt


def _mean_from_sum_count(sum_map: Dict[int, float], cnt_map: Dict[int, int]) -> Dict[int, float]:
    return {k: sum_map[k] / cnt_map[k] for k in sorted(cnt_map.keys()) if cnt_map[k] > 0}


def analyze(path: str) -> Dict[str, object]:
    with open(path, "r", encoding="utf-8") as f:
        hs_sum, hs_cnt, conc_sum, conc_cnt, rnd_sum, rnd_cnt = _accumulate(f)

    summary = {
        "num_records": max(hs_cnt.values()) if hs_cnt else 0,
        "random_expected_pct_mean": (rnd_sum / rnd_cnt) if rnd_cnt else None,
        "layer_hs_pct_mean": _mean_from_sum_count(hs_sum, hs_cnt),
        "layer_conc_ratio_mean": _mean_from_sum_count(conc_sum, conc_cnt),
    }
    return summary


def main():
    parser = argparse.ArgumentParser(description="Aggregate per-layer MoE load metrics from jsonl")
    parser.add_argument("jsonl_path", help="Path to jsonl file with layer stats")
    args = parser.parse_args()

    summary = analyze(args.jsonl_path)
    print(json.dumps(summary, ensure_ascii=True))


if __name__ == "__main__":
    main()
