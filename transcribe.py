"""
Dog vocalization transcription pipeline with bark confidence filtering.

For each sentence in the DB:
  1. Load pre-computed HuBERT features (dog-trained, 768-dim).
  2. Load per-frame bark confidence from ATST-SED predictions.
  3. Identify "dog bark" frames (confidence >= min_confidence).
  4. Within each contiguous bark region, detect phone boundaries via
     adjacent-frame cosine similarity peaks (prominence=0.08, distance=4).
  5. Mean-pool frames within each phone segment → segment embedding.
  6. Optionally auto-estimate K via elbow method (--n_clusters 0).
  7. Fit MiniBatchKMeans incrementally on bark-region segment embeddings (Phase 1).
  8. Save model.joblib, cluster_centers.npy, clustering_metrics.csv.
  9. Re-process → assign cluster labels to bark frames, -1 to non-bark → JSON.

Output JSON: { sentence_id: [int, int, ...] }
  Key   = sentence_id (from DB)
  Value = one label per HuBERT frame (20ms/frame); -1 = non-bark

Usage:
  # fixed K:
  conda run -n dl python transcribe.py --n_clusters 50 --out output/transcription
  # auto K via elbow:
  conda run -n dl python transcribe.py --n_clusters 0  --out output/transcription
"""

import argparse
import json
import logging
import os
import pickle
import time
from typing import Dict, List, Optional, Tuple

import joblib
import mysql.connector
import numpy as np
import pandas as pd
import torch
import torchaudio
from scipy.signal import find_peaks
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger(__name__)

# ── paths ─────────────────────────────────────────────────────────────────────
PRECOMP_FEAT_DIR   = "/path/to/hubert_feat"           # pre-computed HuBERT features dir
DEFAULT_PROB_FRAME = "/path/to/frame_prob.jsonlines"  # ATST-SED per-frame bark confidence
DB_CFG = dict(host="127.0.0.1", port=3306, user="readonly",
              password="YOUR_DB_PASSWORD", database="anilang",
              use_pure=True)

# ── boundary hyperparameters (tuned on 500 dog clips) ─────────────────────────
PROMINENCE  = 0.08
MIN_DIST    = 4          # frames (~80 ms at 50 fps)
HUBERT_SR   = 16_000
HUBERT_FPS  = 50.0       # ~20 ms / frame


# ─────────────────────────────────────────────────────────────────────────────
# Feature extraction
# ─────────────────────────────────────────────────────────────────────────────

def load_hubert_model(device: str):
    bundle = torchaudio.pipelines.HUBERT_BASE
    model  = bundle.get_model().to(device)
    model.eval()
    log.info(f"HuBERT BASE loaded on {device}")
    return model


def extract_hubert_last_layer(
    wav_path: str,
    model: torch.nn.Module,
    device: str,
) -> np.ndarray:
    """Return last-layer HuBERT features (T, 768) as float32."""
    wav, sr = torchaudio.load(wav_path)
    if wav.ndim == 2 and wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != HUBERT_SR:
        wav = torchaudio.functional.resample(wav, sr, HUBERT_SR)
    wav = wav.to(device)
    with torch.inference_mode():
        feats, _ = model.extract_features(wav)
    return feats[-1][0].cpu().numpy().astype(np.float32)


def load_or_extract(
    sentence_id: str,
    audio_path: str,
    feat_cache_dir: str,
    model: Optional[torch.nn.Module],
    device: str,
) -> Optional[np.ndarray]:
    """
    Return (T, 768) features. Priority:
      1. Pre-computed feat dir  (i2r2ck32_45000_hubert_feat)
      2. Local feat cache       (feat_cache_dir / sentence_id.npy)
      3. Extract with HuBERT BASE + save to cache
    Returns None if audio is missing and no cache exists.
    """
    precomp = os.path.join(PRECOMP_FEAT_DIR, sentence_id + ".npy")
    if os.path.exists(precomp):
        return np.load(precomp).astype(np.float32)

    cached = os.path.join(feat_cache_dir, sentence_id + ".npy")
    if os.path.exists(cached):
        return np.load(cached).astype(np.float32)

    if not os.path.exists(audio_path):
        return None
    if model is None:
        return None

    rep = extract_hubert_last_layer(audio_path, model, device)
    np.save(cached, rep)
    return rep


# ─────────────────────────────────────────────────────────────────────────────
# Bark confidence filtering
# ─────────────────────────────────────────────────────────────────────────────

def load_prob_frame(prob_frame_path: str) -> Dict[str, List[float]]:
    """Load bark frame probabilities → {sentence_id: [conf, conf, ...]}."""
    log.info(f"Loading bark confidence scores from {prob_frame_path} ...")
    prob_map: Dict[str, List[float]] = {}
    with open(prob_frame_path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line.strip())
            name = item["audio_name"]
            # Strip .wav extension to match sentence_id
            sid = name[:-4] if name.endswith(".wav") else name
            prob_map[sid] = item["predictions"]
    log.info(f"  Loaded confidence for {len(prob_map):,} files")
    return prob_map


def get_bark_mask(predictions: List[float], T: int, min_conf: float) -> np.ndarray:
    """Project per-frame bark predictions to length T and threshold."""
    P = len(predictions)
    if P == T:
        confs = np.array(predictions, dtype=np.float32)
    else:
        factor = P / T
        indices = np.minimum((np.arange(T) * factor).astype(int), P - 1)
        confs = np.array(predictions, dtype=np.float32)[indices]
    return confs >= min_conf


def get_bark_regions(mask: np.ndarray) -> List[Tuple[int, int]]:
    """Return list of (start, end) for each contiguous bark region."""
    regions = []
    in_region = False
    start = 0
    for i, v in enumerate(mask):
        if v and not in_region:
            start = i
            in_region = True
        elif not v and in_region:
            regions.append((start, i))
            in_region = False
    if in_region:
        regions.append((start, len(mask)))
    return regions


# ─────────────────────────────────────────────────────────────────────────────
# Boundary detection
# ─────────────────────────────────────────────────────────────────────────────

def detect_boundaries(rep: np.ndarray) -> List[int]:
    """Peaks in inverted adjacent-frame cosine similarity → boundary indices."""
    T = rep.shape[0]
    if T < 2:
        return []
    norms = np.linalg.norm(rep, axis=1, keepdims=True)
    norms[norms < 1e-8] = 1.0
    normed = rep / norms
    sim = np.sum(normed[:-1] * normed[1:], axis=1)
    inv = 1.0 - sim
    lo, hi = inv.min(), inv.max()
    if hi > lo:
        inv = (inv - lo) / (hi - lo)
    peaks, _ = find_peaks(inv, prominence=PROMINENCE, distance=MIN_DIST)
    return peaks.tolist()


def pool_segments(rep: np.ndarray, boundaries: List[int]) -> np.ndarray:
    """Mean-pool frames per segment → (n_segments, D)."""
    T = rep.shape[0]
    edges = [0] + [b + 1 for b in boundaries] + [T]
    segs = [rep[s:e].mean(axis=0) for s, e in zip(edges[:-1], edges[1:]) if e > s]
    return np.stack(segs) if segs else rep[:1]


def expand_labels(labels: np.ndarray, boundaries: List[int], T: int) -> np.ndarray:
    """Assign each frame the cluster label of its segment."""
    frame_labels = np.empty(T, dtype=np.int32)
    edges = [0] + [b + 1 for b in boundaries] + [T]
    for seg_idx, (s, e) in enumerate(zip(edges[:-1], edges[1:])):
        frame_labels[s:e] = labels[seg_idx]
    return frame_labels


# ─────────────────────────────────────────────────────────────────────────────
# Per-file processing helpers
# ─────────────────────────────────────────────────────────────────────────────

def compute_bark_regions_and_segs(
    rep: np.ndarray,
    predictions: Optional[List[float]],
    min_conf: float,
) -> Tuple[List[Tuple[int, int, List[int]]], np.ndarray]:
    """
    Returns:
      regions_info: list of (region_start, region_end, internal_boundaries)
        where internal_boundaries are relative to region_start
      all_seg_embs: (N_segments, D) array of pooled embeddings across all bark regions
    If no predictions available, treat entire clip as one bark region.
    """
    T = rep.shape[0]
    if predictions is not None:
        mask = get_bark_mask(predictions, T, min_conf)
        bark_regions = get_bark_regions(mask)
    else:
        bark_regions = [(0, T)]

    regions_info = []
    seg_emb_list = []

    for s, e in bark_regions:
        region_rep = rep[s:e]
        if region_rep.shape[0] < 1:
            continue
        boundaries = detect_boundaries(region_rep)
        seg_embs = pool_segments(region_rep, boundaries)
        regions_info.append((s, e, boundaries))
        seg_emb_list.append(seg_embs)

    all_seg_embs = np.concatenate(seg_emb_list, axis=0) if seg_emb_list else np.empty((0, rep.shape[1]))
    return regions_info, all_seg_embs


def assign_frame_labels(
    regions_info: List[Tuple[int, int, List[int]]],
    kmeans: MiniBatchKMeans,
    rep: np.ndarray,
) -> List[int]:
    """Assign cluster labels to bark frames, -1 to non-bark frames."""
    T = rep.shape[0]
    frame_labels = np.full(T, -1, dtype=np.int32)
    for s, e, boundaries in regions_info:
        region_rep = rep[s:e]
        if region_rep.shape[0] < 1:
            continue
        seg_embs = pool_segments(region_rep, boundaries)
        seg_labels = kmeans.predict(seg_embs)
        region_labels = expand_labels(seg_labels, boundaries, e - s)
        frame_labels[s:e] = region_labels
    return frame_labels.tolist()


# ─────────────────────────────────────────────────────────────────────────────
# K estimation and artifact saving
# ─────────────────────────────────────────────────────────────────────────────

def estimate_k_elbow(X: np.ndarray, k_range=(10, 100)) -> int:
    """Estimate K via second-derivative elbow on MiniBatchKMeans inertia."""
    log.info(f"Elbow method: sweeping K in {k_range} on {len(X):,} sample segments ...")
    inertias = []
    ks = list(range(k_range[0], k_range[1] + 1))
    for k in ks:
        km = MiniBatchKMeans(n_clusters=k, init="k-means++", batch_size=10000,
                             n_init=3, random_state=42)
        km.fit(X)
        inertias.append(km.inertia_)
        log.info(f"  k={k:3d}  inertia={km.inertia_:.2f}")
    deltas        = np.diff(inertias)
    delta_deltas  = np.diff(deltas)
    elbow_idx     = int(np.argmax(delta_deltas)) + k_range[0] + 1  # +1 for double diff
    log.info(f"Elbow at K={elbow_idx}")
    return elbow_idx


def save_artifacts(kmeans: MiniBatchKMeans, features_sample: np.ndarray, out_dir: str):
    """Save model.joblib, cluster_centers.npy, and clustering_metrics.csv."""
    joblib.dump(kmeans, os.path.join(out_dir, "model.joblib"))
    np.save(os.path.join(out_dir, "cluster_centers.npy"), kmeans.cluster_centers_)

    log.info("Computing clustering metrics on sample ...")
    labels = kmeans.predict(features_sample)
    sil  = silhouette_score(features_sample, labels, sample_size=min(10000, len(features_sample)))
    db   = davies_bouldin_score(features_sample, labels)
    ch   = calinski_harabasz_score(features_sample, labels)
    log.info(f"  Silhouette={sil:.4f}  Davies-Bouldin={db:.4f}  Calinski-Harabasz={ch:.2f}")

    metrics_df = pd.DataFrame({
        "Metric": ["n_clusters", "n_samples_fit", "inertia",
                   "silhouette", "davies_bouldin", "calinski_harabasz"],
        "Value":  [kmeans.n_clusters, len(features_sample), kmeans.inertia_,
                   sil, db, ch],
    })
    metrics_df.to_csv(os.path.join(out_dir, "clustering_metrics.csv"), index=False)
    log.info(f"Artifacts saved → {out_dir}/model.joblib, cluster_centers.npy, clustering_metrics.csv")


# ─────────────────────────────────────────────────────────────────────────────
# Database
# ─────────────────────────────────────────────────────────────────────────────

def fetch_sentences(max_files: int = 0) -> List[Tuple[str, str]]:
    """Stream sentences from DB. max_files=0 means all."""
    log.info("Fetching sentence list from DB ...")
    conn = mysql.connector.connect(**DB_CFG)
    cur  = conn.cursor(buffered=False)
    if max_files > 0:
        cur.execute(f"SELECT sentence_id, audio_path FROM sentences LIMIT {max_files}")
    else:
        cur.execute("SELECT sentence_id, audio_path FROM sentences")
    rows = list(cur)
    cur.close()
    conn.close()
    log.info(f"  {len(rows):,} sentences fetched")
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_clusters",     type=int,   default=50,
                        help="Number of clusters; 0 = auto via elbow method")
    parser.add_argument("--elbow_k_min",   type=int,   default=10)
    parser.add_argument("--elbow_k_max",   type=int,   default=100)
    parser.add_argument("--elbow_sample",  type=int,   default=200_000,
                        help="Max segments sampled for elbow / metrics")
    parser.add_argument("--batch_size",    type=int,   default=512,
                        help="Files per MiniBatchKMeans partial_fit call")
    parser.add_argument("--out",            type=str,   default="output/transcription")
    parser.add_argument("--device",         type=str,   default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--resume",         action="store_true",
                        help="Skip Phase 1 if kmeans checkpoint exists")
    parser.add_argument("--log_every",      type=int,   default=5000)
    parser.add_argument("--max_files",      type=int,   default=0,
                        help="Limit to first N sentences (0 = all)")
    parser.add_argument("--prob_frame",     type=str,   default=DEFAULT_PROB_FRAME,
                        help="Path to ATST-SED frame_prob_result_v2.jsonlines")
    parser.add_argument("--min_confidence", type=float, default=0.67,
                        help="Bark confidence threshold (default: 0.67)")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    feat_cache_dir = os.path.join(args.out, "feat_cache")
    os.makedirs(feat_cache_dir, exist_ok=True)

    ckpt_kmeans  = os.path.join(args.out, "kmeans.pkl")
    ckpt_regions = os.path.join(args.out, "regions_cache.pkl")
    out_json     = os.path.join(args.out, "transcription.json")

    sentences  = fetch_sentences(max_files=args.max_files)
    prob_map   = load_prob_frame(args.prob_frame) if os.path.exists(args.prob_frame) else {}
    if not prob_map:
        log.warning(f"No bark confidence file found at {args.prob_frame}; all frames treated as bark.")

    # ── Phase 1: fit MiniBatchKMeans ──────────────────────────────────────────
    if args.resume and os.path.exists(ckpt_kmeans) and os.path.exists(ckpt_regions):
        log.info("Resuming: loading KMeans + regions cache from checkpoint ...")
        with open(ckpt_kmeans,  "rb") as f: kmeans = pickle.load(f)
        with open(ckpt_regions, "rb") as f: regions_cache = pickle.load(f)
        log.info(f"  Loaded regions cache for {len(regions_cache):,} sentences")
    else:
        # ── Collect bark-segment embeddings (streaming) ──────────────────────
        log.info("Phase 1a — collecting bark-region segment embeddings ...")
        regions_cache: Dict[str, List[Tuple[int, int, List[int]]]] = {}
        model = None
        seg_buffer:  List[np.ndarray] = []   # for partial_fit batches
        sample_pool: List[np.ndarray] = []   # reservoir for elbow / metrics
        n_sample = 0
        n_done = 0
        n_skipped = 0
        t0 = time.time()

        for sid, audio_path in sentences:
            rep = load_or_extract(sid, audio_path, feat_cache_dir, model, args.device)

            if rep is None:
                if model is None:
                    log.info("Loading HuBERT BASE for on-the-fly extraction ...")
                    model = load_hubert_model(args.device)
                rep = load_or_extract(sid, audio_path, feat_cache_dir, model, args.device)

            if rep is None or rep.shape[0] < 2:
                n_skipped += 1
                regions_cache[sid] = []
                n_done += 1
                continue

            predictions = prob_map.get(sid, None)
            regions_info, seg_embs = compute_bark_regions_and_segs(
                rep, predictions, args.min_confidence
            )
            regions_cache[sid] = regions_info

            if seg_embs.shape[0] > 0:
                seg_buffer.append(seg_embs)
                # reservoir sample for elbow / metrics
                if n_sample < args.elbow_sample:
                    sample_pool.append(seg_embs)
                    n_sample += seg_embs.shape[0]

            n_done += 1
            if n_done % args.log_every == 0:
                elapsed = time.time() - t0
                rate = n_done / elapsed
                eta  = (len(sentences) - n_done) / rate if rate > 0 else 0
                log.info(f"  Phase 1a: {n_done:,}/{len(sentences):,}  "
                         f"skipped={n_skipped}  "
                         f"{rate:.0f} files/s  ETA {eta/60:.1f} min")

        log.info(f"Phase 1a done. Collected {n_sample:,} bark segments, skipped {n_skipped:,} files.")

        # ── Determine K ───────────────────────────────────────────────────────
        sample_arr = np.concatenate(sample_pool, axis=0)[:args.elbow_sample]

        if args.n_clusters == 0:
            n_clusters = estimate_k_elbow(
                sample_arr, k_range=(args.elbow_k_min, args.elbow_k_max)
            )
            log.info(f"Auto-selected K={n_clusters}")
        else:
            n_clusters = args.n_clusters
            log.info(f"Using specified K={n_clusters}")

        # ── Fit MiniBatchKMeans ───────────────────────────────────────────────
        log.info(f"Phase 1b — fitting MiniBatchKMeans (k={n_clusters}) ...")
        kmeans = MiniBatchKMeans(
            n_clusters=n_clusters,
            init="k-means++",
            batch_size=4096,
            n_init=5,
            random_state=42,
            verbose=0,
        )
        for i in range(0, len(seg_buffer), args.batch_size):
            batch = np.concatenate(seg_buffer[i:i + args.batch_size], axis=0)
            kmeans.partial_fit(batch)

        log.info(f"Phase 1b done. Inertia={kmeans.inertia_:.2f}")

        # ── Save artifacts ────────────────────────────────────────────────────
        save_artifacts(kmeans, sample_arr, args.out)
        with open(ckpt_kmeans,  "wb") as f: pickle.dump(kmeans, f)
        with open(ckpt_regions, "wb") as f: pickle.dump(regions_cache, f)
        log.info(f"Checkpoint saved → {ckpt_kmeans}")

    # ── Phase 2: assign frame-level labels ────────────────────────────────────
    log.info("Phase 2 — assigning frame-level cluster labels ...")
    transcription: Dict[str, List[int]] = {}
    n_done = 0
    n_skipped = 0
    t0 = time.time()
    model = None

    for sid, audio_path in sentences:
        regions_info = regions_cache.get(sid, None)

        rep = load_or_extract(sid, audio_path, feat_cache_dir, model, args.device)
        if rep is None:
            if model is None:
                log.info("Loading HuBERT BASE for on-the-fly extraction ...")
                model = load_hubert_model(args.device)
            rep = load_or_extract(sid, audio_path, feat_cache_dir, model, args.device)

        if rep is None or rep.shape[0] < 1:
            n_skipped += 1
            n_done += 1
            continue

        if regions_info is None:
            predictions = prob_map.get(sid, None)
            regions_info, _ = compute_bark_regions_and_segs(
                rep, predictions, args.min_confidence
            )

        transcription[sid] = assign_frame_labels(regions_info, kmeans, rep)

        n_done += 1
        if n_done % args.log_every == 0:
            elapsed = time.time() - t0
            rate = n_done / elapsed
            eta  = (len(sentences) - n_done) / rate if rate > 0 else 0
            log.info(f"  Phase 2: {n_done:,}/{len(sentences):,}  "
                     f"skipped={n_skipped}  "
                     f"{rate:.0f} files/s  ETA {eta/60:.1f} min")

    log.info(f"Phase 2 done. {len(transcription):,} sentences transcribed, "
             f"{n_skipped:,} skipped.")
    log.info(f"Writing {out_json} ...")
    with open(out_json, "w") as f:
        json.dump(transcription, f)
    log.info(f"Done. Output: {out_json}")


if __name__ == "__main__":
    main()
