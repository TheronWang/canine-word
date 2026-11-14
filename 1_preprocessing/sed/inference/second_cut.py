import argparse
import bisect
import json
import logging
import os
import sys
from datetime import datetime

import librosa
import numpy as np
import torch
import yaml
from torch import nn
from tqdm import tqdm

sys.path.append("../")
sys.path.append("../../")
from util import set_seed, log_init
from desed_task.dataio.datasets_atst_sed import (
    SEDTransform,
    ATSTTransform,
    read_audio,
)
from desed_task.utils.scaler import TorchScaler
from torchaudio.transforms import AmplitudeToDB, MelSpectrogram
from desed_task.utils.encoder import ManyHotEncoder
from collections import OrderedDict
from desed_task.nnet.CRNN_beats import CRNN
import pytorch_lightning as pl

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

classes_labels = OrderedDict({
    "Dog": 0,
})

"""
conda activate /home/xingyuan/anaconda3/envs/atst_sed_dasheng


python second_cut.py \ --input_root /data/dog_age_classification/dataset/age/pre_process_dataset/dataset/first_cut/separation \ --output_path /data/dog_age_classification/dataset/age/pre_process_dataset/dataset/second_cut/dog_age_44k_thr0.05_offset1_0.5.jsonlines
"""


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_root", type=str, required=True, help="The root of input"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="The path of output (JSONLINES FILE)",
    )
    parser.add_argument(
        "--log_root",
        type=str,
        required=False,
        # default="/data/dog_age_classification/dataset/age/pre_process_dataset/logs/v2/log",
        default="/data/dog_age_classification/dataset/age/pre_process_dataset/logs/v2/log",
        help="The root of log",
    )
    parser.add_argument(
        "--config_path",
        type=str,
        required=False,
        default="../train/frozen_udesed.yaml",
        help="The path of config",
    )
    parser.add_argument(
        "--model_mode",
        type=str,
        required=False,
        default="student",
        help="The model mode",
    )

    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=False,
        default="/data/xingyuan/animal_language/v2/out/second_cut_model/beats/stage1/version_1/checkpoints/epoch=109-step=6160.ckpt",
        help="The path of checkpoint",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        required=False,
        default=0.671329140663147,
        help="The threshold",
    )
    parser.add_argument(
        "--device",
        type=int,
        required=False,
        default=0,
        help="The number of GPU (-1 is CPU)",
    )
    parser.add_argument(
        "--seed", type=int, required=False, default=42, help="The seed"
    )
    return parser.parse_args()


# ------------------------------------ inference


class BEATSSEDFeatureExtractor(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.sed_feat_extractor = SEDTransform(config["feats"])
        self.scaler = TorchScaler(
            "instance",
            config["scaler"]["normtype"],
            config["scaler"]["dims"],
        )
        self.sed_feat_extractor.transform.to("cuda:0")

    def take_log(self, mels):
        amp_to_db = AmplitudeToDB(stype="amplitude")
        amp_to_db.amin = 1e-5  # amin= 1e-5 as in librosa
        return amp_to_db(mels).clamp(
            min=-50, max=80
        )  # clamp to reproduce old code

    def forward(self, mixture):
        mixture = mixture.unsqueeze(0)  # fake batch size
        sed_feats = self.sed_feat_extractor(mixture)
        sed_feats = self.scaler(self.take_log(sed_feats))
        return sed_feats, mixture


class BEATSSEDInferencer(nn.Module):
    """Inference module for ATST-SED"""

    def __init__(
        self,
        config,
        encoder,
        overlap_dur=5,
    ):
        super().__init__()

        self.config = config
        self.label_encoder = encoder
        self.feature_extractor = BEATSSEDFeatureExtractor(config)

        # Initial parameters
        self.audio_dur = 10
        self.overlap_dur = overlap_dur
        self.fs = config["data"]["fs"]

        # Unfolder for splitting audio into chunks
        self.unfolder = nn.Unfold(
            kernel_size=(self.fs * self.audio_dur, 1),
            stride=(self.fs * self.overlap_dur, 1),
        )

        # self.hard_threshold = [hard_threshold] * len(self.label_encoder.labels) if not isinstance(hard_threshold, list) else hard_threshold

        self.hard_threshold = None

    def get_logmel(self, wav_file):
        mixture, onset_s, offset_s, padded_indx = read_audio(
            wav_file, False, False, None
        )
        sed_feats, _ = self.feature_extractor(mixture)
        return sed_feats[0].detach().cpu().numpy()

    def forward(self, mixture, model):

        # split wav into chunks with overlap
        if mixture.shape[0] <= (self.audio_dur * self.fs):
            inference_chunks = [mixture]
            padding_frames = 0
            mixture_pad = mixture.clone()
        else:
            # pad the mixtures
            mixture = mixture.unsqueeze(0).unsqueeze(0).unsqueeze(-1)
            total_chunks = (
                mixture.numel()
                - ((self.audio_dur - self.overlap_dur) * self.fs)
            ) // (self.overlap_dur * self.fs) + 1
            total_length = (
                total_chunks * self.overlap_dur * self.fs
                + (self.audio_dur - self.overlap_dur) * self.fs
            )
            mixture_pad = torch.nn.functional.pad(
                mixture, (0, 0, 0, total_length - mixture.numel())
            )
            padding_frames = self.time2frame(total_length - mixture.numel())
            inference_chunks = self.unfolder(mixture_pad)
            inference_chunks = inference_chunks.squeeze(0).T

            if (total_length - mixture.numel()) > 0:
                padding_frames = max(padding_frames, 1)

        # inference result for each chunk
        sed_results = []
        for chunk in inference_chunks:
            sed_feats, mixture = self.feature_extractor(chunk)
            # sed_feats = sed_feats.transpose(1, 2)
            chunk_result, _ = model(sed_feats, mixture)
            sed_results.append(chunk_result)

        if self.hard_threshold is None:
            # return sed_results  # If no threshold given, return soft results and you can customize the threshold later
            chunk_scores = [
                self.get_post_scores(r.detach().cpu()) for r in sed_results
            ]
            return self.unify_scores(
                chunk_scores=chunk_scores,
                total_frames=self.time2frame(mixture_pad.numel()),
                padding_frames=padding_frames,
            )
        else:
            assert False

    def get_post_scores(self, strong_preds):
        strong_preds = strong_preds[0]  # only support single input (bsz=1)
        # smoothed_preds = []
        # for score, fil_val in zip(strong_preds, self.config["training"]["median_window"]):
        #     score = scipy.ndimage.filters.median_filter(score[:, np.newaxis], (fil_val, 1))
        #     smoothed_preds.append(score)
        # smoothed_preds = np.concatenate(smoothed_preds, axis=1)
        # return smoothed_preds.T

        return strong_preds.numpy()

    def unify_scores(self, chunk_scores, total_frames, padding_frames):
        # TODO: 只能用在一个类
        C, T = chunk_scores[0].shape
        if len(chunk_scores) == 1:
            return chunk_scores[0][0]
        else:
            decisions = np.zeros((C, total_frames))
            hop_frame = self.time2frame(self.overlap_dur * self.fs)
            decisions[:, 0 * hop_frame : 0 * hop_frame + T] += chunk_scores[0]
            # for i in range(1, len(chunk_scores)):
            #     decisions[:, i * hop_frame: i * hop_frame + T] += chunk_scores[i]
            #     decisions[:, i * hop_frame: i * hop_frame + T - hop_frame] *= 0.5

            for i in range(1, len(chunk_scores)):
                assert (
                    decisions[:, i * hop_frame : i * hop_frame + T].shape
                    == chunk_scores[i].shape
                )
                decisions[:, i * hop_frame : i * hop_frame + T] = np.maximum(
                    chunk_scores[i],
                    decisions[:, i * hop_frame : i * hop_frame + T],
                )
            return decisions[0, :-padding_frames]

    def time2frame(self, time):
        return int(
            int((time / self.label_encoder.frame_hop))
            / self.label_encoder.net_pooling
        )


# --------------------------------------- inference


def find_contiguous_regions(activity_array):
    """Find contiguous regions from bool valued numpy.array.
    Copy of https://dcase-repo.github.io/dcase_util/_modules/dcase_util/data/decisions.html#DecisionEncoder

    Reason is:
    1. This does not belong to a class necessarily
    2. Import DecisionEncoder requires sndfile over some other imports..which causes some problems on clusters

    """
    # Find the changes in the activity_array
    change_indices = np.logical_xor(
        activity_array[1:], activity_array[:-1]
    ).nonzero()[0]

    # Shift change_index with one, focus on frame after the change.
    change_indices += 1

    if activity_array[0]:
        # If the first element of activity_array is True add 0 at the beginning
        change_indices = np.r_[0, change_indices]

    if activity_array[-1]:
        # If the last element of activity_array is True, add the length of the array
        change_indices = np.r_[change_indices, activity_array.size]

    # Reshape the result into two columns
    return change_indices.reshape((-1, 2))


def interval_union(clips, interval_time=0.5):
    if len(clips) < 2:
        return clips
    clips.sort(key=lambda c: c[0])
    start = clips[0][0]
    end = clips[0][1]

    new_clips = []
    for i in range(1, len(clips)):
        if (clips[i][0] - end) < interval_time:
            end = max(clips[i][1], end)
        else:
            new_clips.append([start, end])
            start = clips[i][0]
            end = clips[i][1]
    new_clips.append([start, end])
    return new_clips


if __name__ == "__main__":
    args = get_args()
    set_seed(seed=args.seed, speed_up=False)
    pl.seed_everything(args.seed, workers=True)
    input_root = args.input_root
    output_path = args.output_path
    log_root = args.log_root
    config_path = args.config_path
    checkpoint_path = args.checkpoint_path
    log_path = os.path.join(
        log_root,
        f'[second_cut]_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.log',
    )

    assert os.path.exists(
        input_root
    ), f"File or folder not found: {input_root}"
    assert os.path.exists(
        config_path
    ), f"File or folder not found: {input_root}"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    os.makedirs(log_root, exist_ok=True)
    assert os.path.exists(
        checkpoint_path
    ), f"File or folder not found: {checkpoint_path}"

    log_init(log_path)
    logging.info(args)

    torch.cuda.set_device(args.device)

    input_datas = []
    for file in os.listdir(input_root):
        input_datas.append(os.path.join(input_root, file))
    input_audio_paths = []
    if os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as file:
            output_datas = [
                json.loads(line.strip())
                for line in file.readlines()
                if len(line.strip()) > 0
            ]
        output_datas = [od["path"] for od in output_datas]
        output_datas.sort()
        for id in input_datas:
            ind = bisect.bisect_left(output_datas, id)
            if not (
                0 <= ind < len(output_datas) and (output_datas[ind] == id)
            ):
                input_audio_paths.append(id)
    else:
        input_audio_paths = input_datas
    input_audio_paths = list(set(input_audio_paths))
    input_audio_paths.sort()
    logging.info(
        f"Input data loaded. Will cut {len(input_audio_paths)} audios"
        f" (total={len(input_datas)})."
    )

    # 推理相关
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    encoder = ManyHotEncoder(
        list(classes_labels.keys()),
        audio_len=config["data"]["audio_max_len"],
        frame_len=config["feats"]["n_filters"],
        frame_hop=config["feats"]["hop_length"],
        net_pooling=config["data"]["net_subsample"],
        fs=config["data"]["fs"],
    )
    inference_model = BEATSSEDInferencer(config=config, encoder=encoder)

    # model
    model = CRNN(
        unfreeze_atst_layer=config["opt"]["tfm_trainable_layers"],
        **config["net"],
        model_init=checkpoint_path,
        beats_init=config["ultra"]["beats_init"],
        mode=args.model_mode,
    )
    model.eval()
    device = torch.device(f"cuda:{args.device}")
    model = model.to(device)

    frames_per_second = config["data"]["fs"] / (
        config["data"]["net_subsample"] * config["feats"]["hop_length"]
    )
    min_length = int(10.0 * 16000)
    for path in tqdm(input_audio_paths):
        mixture, onset_s, offset_s, padded_indx = read_audio(
            path, False, False, None
        )
        # duration = mixture.shape[0] / 16000.
        if mixture.shape[0] < min_length:
            mixture = torch.nn.functional.pad(
                mixture,
                (0, min_length - mixture.shape[0]),
                mode="constant",
                value=0.0,
            )
            assert mixture.shape[0] == min_length

        wav, sr = librosa.load(path=path, sr=None, mono=True)
        duration = wav.shape[0] / sr
        mixture = mixture.to(device)
        with torch.no_grad():
            model.eval()
            model.to(device)
            sed_results = inference_model(mixture=mixture, model=model)

        filtered_dog_framewise_output = np.where(
            sed_results >= args.threshold, 1, 0
        )
        continue_regions = find_contiguous_regions(
            filtered_dog_framewise_output
        )
        continue_regions = continue_regions / frames_per_second
        continue_regions = np.maximum(continue_regions, 0.0)
        continue_regions = np.minimum(continue_regions, duration)
        clips = continue_regions.tolist()
        clips.sort(key=lambda c: c[0])
        raw_clips = interval_union(clips=clips, interval_time=0.5)

        result_clips = []
        for i in range(len(raw_clips)):
            result_clips.append({
                'id': str(i).zfill(5),
                'start': raw_clips[i][0],
                'end': raw_clips[i][1]
            })
        if len(result_clips) > 0:
            result = {
                "name": os.path.basename(path)[: -len(".wav")],
                "path": path,
                "clips": result_clips,
            }
            with open(output_path, "a+", encoding="utf-8") as file:
                file.write(json.dumps(result, ensure_ascii=False))
                file.write("\n")

    logging.info(f"Successfully cut {len(input_audio_paths)} audios.")
