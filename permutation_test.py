"""
Permutation Control Experiment for Word Finding (NPMI) — EM boundary model.

Procedure:
  1. Run EM boundary model once on real data → fixed per-clip segmentation
  2. Compute real NPMI from that segmentation
  3. For K permutations: shuffle keyword sets across clips (preserves marginals),
     recompute NPMI on the FIXED segmentation
  4. Report real NPMI, mean/std permuted NPMI, z-score, empirical p-value

Usage:
  python permutation_test.py --transcription <path> --keywords <path> [--n_perm 10] [--output ./output]
"""

import argparse
import json
import logging
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from word_finding import WordFinder

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

def compute_npmi(joint: int, ngram_cnt: int, keyword_cnt: int, total: int) -> float:
    if joint == 0:
        return -1.0
    p_joint   = joint      / total
    p_ngram   = ngram_cnt  / total
    p_keyword = keyword_cnt / total
    pmi  = math.log(p_joint / (p_ngram * p_keyword))
    npmi = pmi / -math.log(p_joint)
    return npmi


def main():
    parser = argparse.ArgumentParser(description="Permutation significance test for word finding")
    parser.add_argument("--transcription", required=True, help="Path to transcription JSON")
    parser.add_argument("--keywords",      required=True, help="Path to keywords JSON")
    parser.add_argument("--output",        default="./output/permutation_test")
    parser.add_argument("--n_perm",        type=int,   default=10)
    parser.add_argument("--seed",          type=int,   default=42)
    parser.add_argument("--ngram_max",     type=int,   default=10)
    parser.add_argument("--em_iters",      type=int,   default=10)
    parser.add_argument("--min_support",   type=int,   default=50)
    parser.add_argument("--min_npmi",      type=float, default=0.2)
    args = parser.parse_args()

    OUTPUT_DIR = Path(args.output)
    rng = np.random.default_rng(args.seed)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load data ──────────────────────────────────────────────────────────────
    log.info("Loading transcription...")
    with open(args.transcription) as f:
        transcription = json.load(f)

    log.info("Loading keywords...")
    with open(args.keywords) as f:
        keywords_raw = json.load(f)

    transcription = {k: v for k, v in transcription.items() if k in keywords_raw}
    log.info(f"Clips with both transcription and keywords: {len(transcription):,}")

    # ── Run EM boundary model once on real data ────────────────────────────────
    log.info("Running EM boundary model on real data (this takes a while)...")
    wf = WordFinder(transcription, keywords_raw)
    wf.set_parser(
        "boundary_prob",
        ngram_range=(1, args.ngram_max),
        max_iterations=args.em_iters,
        use_keyword_context=True,
    )
    wf.set_min_support(args.min_support)
    wf.set_min_eval_score(args.min_npmi)
    wf.set_evaluator("npmi")
    wf.parse()

    # ── Extract real top pairs from NPMI evaluator ─────────────────────────────
    real_pairs = wf.get_words()   # returns list of dicts without saving
    if not real_pairs:
        log.warning("No valid ngram-keyword pairs found above thresholds. Exiting.")
        return

    log.info(f"Real top pairs found: {len(real_pairs)}")
    for r in real_pairs:
        log.info(f"  {r['ngram']} ↔ {r['keyword']}  NPMI={r['npmi']:.4f}  support={r['support']}")

    # ── Extract per-clip ngram sets from EM segmentation ──────────────────────
    # sentences are already parsed — pull ngram sets from parsed_result
    clip_ids      = list(wf.sentences.keys())
    clip_keywords = [wf.sentences[cid].keywords for cid in clip_ids]
    N             = len(clip_ids)

    # For each top pair, build boolean arrays over clips
    unique_ngrams   = list({tuple(r["ngram"]) for r in real_pairs})
    unique_keywords = list({r["keyword"]      for r in real_pairs})

    ng_idx = {ng: i for i, ng in enumerate(unique_ngrams)}
    kw_idx = {kw: i for i, kw in enumerate(unique_keywords)}

    log.info("Building per-clip ngram presence arrays...")
    ngram_arr   = np.zeros((len(unique_ngrams),   N), dtype=bool)
    keyword_arr = np.zeros((len(unique_keywords), N), dtype=bool)

    for i, cid in enumerate(clip_ids):
        sentence = wf.sentences[cid]
        # Collect all ngrams produced by EM parse for this clip
        clip_ngrams = set()
        for seg_ngrams in sentence.parsed_result:
            clip_ngrams.update(seg_ngrams)

        for ng in unique_ngrams:
            if ng in clip_ngrams:
                ngram_arr[ng_idx[ng], i] = True
        for kw in unique_keywords:
            if kw in sentence.keywords:
                keyword_arr[kw_idx[kw], i] = True

    ngram_counts   = ngram_arr.sum(axis=1)
    keyword_counts = keyword_arr.sum(axis=1)

    # ── Permutation loop ───────────────────────────────────────────────────────
    log.info(f"Running {args.n_perm} permutations (keywords shuffled, EM segmentation fixed)...")
    perm_npmi = defaultdict(list)

    for k in range(args.n_perm):
        log.info(f"  Permutation {k+1}/{args.n_perm}")
        perm = rng.permutation(N)
        perm_kw_arr = keyword_arr[:, perm]

        for r in real_pairs:
            ng = tuple(r["ngram"])
            kw = r["keyword"]
            joint = int(np.sum(ngram_arr[ng_idx[ng]] & perm_kw_arr[kw_idx[kw]]))
            npmi  = compute_npmi(joint, int(ngram_counts[ng_idx[ng]]),
                                 int(keyword_counts[kw_idx[kw]]), N)
            perm_npmi[(ng, kw)].append(npmi)

    log.info("Permutations complete.")

    # ── Report ─────────────────────────────────────────────────────────────────
    rows = []
    header = ["ngram", "keyword", "real_npmi", "real_support", "ngram_cnt", "keyword_cnt",
              "mean_perm_npmi", "std_perm_npmi", "z_score", "p_value", "n_perm"]

    print("\n" + "=" * 115)
    print(f"{'Ngram':<22} {'Keyword':<20} {'Real':>7} {'Support':>8} {'MeanPerm':>10} {'StdPerm':>9} {'Z':>7} {'p-val':>8}")
    print("=" * 115)

    for r in real_pairs:
        ng       = tuple(r["ngram"])
        kw       = r["keyword"]
        real_val = r["npmi"]
        support  = r["support"]
        ng_cnt   = r["ngram_cnt"]
        kw_cnt   = r["keyword_cnt"]

        perm_vals = np.array(perm_npmi[(ng, kw)])
        mean_p = float(np.mean(perm_vals))
        std_p  = float(np.std(perm_vals))
        z      = (real_val - mean_p) / std_p if std_p > 0 else float("inf")
        p_val  = float(np.mean(perm_vals >= real_val))

        ng_str = str(list(ng))
        print(f"{ng_str:<22} {kw:<20} {real_val:>7.4f} {support:>8} {mean_p:>10.4f} {std_p:>9.4f} {z:>7.2f} {p_val:>8.4f}")

        rows.append({
            "ngram": ng_str, "keyword": kw,
            "real_npmi": round(real_val, 6), "real_support": support,
            "ngram_cnt": ng_cnt, "keyword_cnt": kw_cnt,
            "mean_perm_npmi": round(mean_p, 6), "std_perm_npmi": round(std_p, 6),
            "z_score": round(z, 4), "p_value": round(p_val, 4),
            "n_perm": args.n_perm,
        })

    print("=" * 115)

    out_json = OUTPUT_DIR / f"permutation_em_results_K{args.n_perm}.json"
    out_csv  = OUTPUT_DIR / f"permutation_em_results_K{args.n_perm}.csv"

    with open(out_json, "w") as f:
        json.dump(rows, f, indent=2)
    log.info(f"Saved JSON to {out_json}")

    with open(out_csv, "w") as f:
        f.write(",".join(header) + "\n")
        for row in rows:
            f.write(",".join(str(row[h]) for h in header) + "\n")
    log.info(f"Saved CSV to {out_csv}")

    sig = [r for r in rows if r["p_value"] < 0.05]
    log.info(f"Significant pairs (p < 0.05): {len(sig)} / {len(rows)}")


if __name__ == "__main__":
    main()
