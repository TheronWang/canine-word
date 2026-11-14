"""Download model weights and data for canine-word (ACL 2025) from the Hugging Face Hub.

    pip install huggingface_hub
    python download.py                 # everything (about 1.3 GB)
    python download.py --models        # model weights only
    python download.py --data          # transcriptions, segments, video durations only
    python download.py --models --only dog_hubert_stage2_130k.pt kmeans_layer11_k250.pkl

Models go to ./checkpoints and data to ./data (both are git-ignored).
"""
import argparse
import os

from huggingface_hub import hf_hub_download, snapshot_download

MODEL_REPO = "ArlingtonCL2/canine-phonetic-models"
DATA_REPO = "ArlingtonCL2/canine-phonetic-data"

HERE = os.path.dirname(os.path.abspath(__file__))

MODEL_FILES = {
    "beats_sed_dog.ckpt": "BEATs sound event detector for cutting vocal segments (step 1)",
    "beats_sed_dog_config.yaml": "config of the detector, for training and inference (step 1)",
    "dog_hubert_stage2_130k.pt": "Dog-HuBERT after stage 2, 130k updates (step 2)",
    "kmeans_layer11_k250.pkl": "k-means over layer-11 features, 250 initial vocal units (step 2)",
    "stage1_targets_mfcc_k100.pkl": "MFCC k-means, stage-1 HuBERT targets (step 2, training only)",
    "stage2_targets_layer6_k200.pkl": "layer-6 k-means, stage-2 HuBERT targets (step 2, training only)",
}


def download_models(only=None):
    target = os.path.join(HERE, "checkpoints")
    for name, desc in MODEL_FILES.items():
        if only and name not in only:
            continue
        print(f"[models] {name}: {desc}")
        hf_hub_download(MODEL_REPO, name, local_dir=target)
    return target


def download_data():
    target = os.path.join(HERE, "data")
    print(f"[data] {DATA_REPO}")
    snapshot_download(DATA_REPO, repo_type="dataset", local_dir=target)
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", action="store_true", help="download model weights")
    parser.add_argument("--data", action="store_true", help="download transcriptions, segments and video durations")
    parser.add_argument("--only", nargs="+", choices=sorted(MODEL_FILES), help="with --models: only these files")
    args = parser.parse_args()

    both = not (args.models or args.data)
    if args.models or both:
        print("saved to", download_models(args.only))
    if args.data or both:
        print("saved to", download_data())


if __name__ == "__main__":
    main()
