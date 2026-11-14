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

sys.path.append('../')
sys.path.append('../../')
from util import set_seed, log_init
from desed_task.dataio.datasets_atst_sed import SEDTransform, ATSTTransform, read_audio
from desed_task.utils.scaler import TorchScaler
from torchaudio.transforms import AmplitudeToDB, MelSpectrogram
from desed_task.utils.encoder import ManyHotEncoder
from collections import OrderedDict
from desed_task.nnet.CRNN_beats import CRNN
from evaluate2 import get_seds_pred, get_other_metrics

classes_labels = OrderedDict(
    {
        "Dog": 0,
    }
)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_root', type=str, required=True,
                        help='The root of input')
    parser.add_argument('--output_path', type=str, required=True,
                        help='The path of output (JSONLINES FILE)')
    parser.add_argument('--log_root', type=str, required=True,
                        help='The root of log')
    parser.add_argument('--config_path', type=str, required=True,
                        help='The path of config')
    parser.add_argument('--model_mode', type=str, required=True,
                        help='The model mode')

    parser.add_argument('--checkpoint_path', type=str, required=False,
                        default='',
                        help='The path of checkpoint')
    parser.add_argument('--threshold', type=float, required=True,
                        help='The threshold')
    parser.add_argument('--device', type=int, required=False,
                        default=0,
                        help='The number of GPU (-1 is CPU)')
    parser.add_argument('--seed', type=int, required=False,
                        default=42,
                        help='The seed')
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
        self.sed_feat_extractor.transform.to('cuda:0')

    def take_log(self, mels):
        amp_to_db = AmplitudeToDB(stype="amplitude")
        amp_to_db.amin = 1e-5  # amin= 1e-5 as in librosa
        return amp_to_db(mels).clamp(min=-50, max=80)  # clamp to reproduce old code

    def forward(self, mixture):
        mixture = mixture.unsqueeze(0)  # fake batch size
        sed_feats = self.sed_feat_extractor(mixture)
        sed_feats = self.scaler(self.take_log(sed_feats))
        return sed_feats, mixture


class BEATSSEDInferencer(nn.Module):
    """Inference module for ATST-SED
    """

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
        self.unfolder = nn.Unfold(kernel_size=(self.fs * self.audio_dur, 1), stride=(self.fs * self.overlap_dur, 1))

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
            total_chunks = (mixture.numel() - ((self.audio_dur - self.overlap_dur) * self.fs)) // (
                    self.overlap_dur * self.fs) + 1
            total_length = total_chunks * self.overlap_dur * self.fs + (self.audio_dur - self.overlap_dur) * self.fs
            mixture_pad = torch.nn.functional.pad(mixture, (0, 0, 0, total_length - mixture.numel()))
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
            chunk_scores = [self.get_post_scores(r.detach().cpu()) for r in sed_results]
            return self.unify_scores(chunk_scores=chunk_scores,
                                     total_frames=self.time2frame(mixture_pad.numel()),
                                     padding_frames=padding_frames)
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
            decisions[:, 0 * hop_frame: 0 * hop_frame + T] += chunk_scores[0]
            # for i in range(1, len(chunk_scores)):
            #     decisions[:, i * hop_frame: i * hop_frame + T] += chunk_scores[i]
            #     decisions[:, i * hop_frame: i * hop_frame + T - hop_frame] *= 0.5

            for i in range(1, len(chunk_scores)):
                assert decisions[:, i * hop_frame: i * hop_frame + T].shape == chunk_scores[i].shape
                decisions[:, i * hop_frame: i * hop_frame + T] = np.maximum(chunk_scores[i], decisions[:,
                                                                                             i * hop_frame: i * hop_frame + T])
            return decisions[0, : -padding_frames]

    def time2frame(self, time):
        return int(int((time / self.label_encoder.frame_hop)) / self.label_encoder.net_pooling)

# --------------------------------------- inference

def get_sed_duration(path):
    sed_durations = {}
    with open(path, 'r') as file:
        duration_data = [line.strip().split() for line in file.readlines() if len(line.strip()) > 0]
    for dd in duration_data[1:]:
        assert dd[0] not in sed_durations.keys()
        sed_durations[dd[0]] = float(dd[1])
    return sed_durations


def interval_union(clips, interval_time=1e-3):
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
            new_clips.append([start, end, 'Dog'])
            start = clips[i][0]
            end = clips[i][1]
    new_clips.append([start, end, 'Dog'])
    return new_clips


def get_sed_ground_truth(path, audio_root, test_positive_labels):
    seds_ground_truth = {f: [] for f in os.listdir(audio_root)}
    with open(path, 'r') as file:
        gt_data = [line.strip().split() for line in file.readlines() if len(line.strip()) > 0]
    for gd in gt_data[1:]:
        if gd[3] in test_positive_labels:
            seds_ground_truth[gd[0]].append([float(gd[1]), float(gd[2]), 'Dog'])
    for key in seds_ground_truth.keys():
        seds_ground_truth[key] = interval_union(seds_ground_truth[key])
    return seds_ground_truth

def get_test_dataset():
    test_positive_labels = 'Dog,dog,dog_dog'.split(',')
    keys = ['chihuahua', 'husky', 'shiba_inu', 'pitbull', 'labrador', 'german_shepherd']
    tsv_root = '/home/xingyuan/animal_language2/data/dog_sed_dataset2/strong_valid_labels_tsv'
    audio_root = '/home/xingyuan/animal_language2/data/dog_sed_dataset2/strong_valid_audios_16k'

    seds_duration_dict = {key: {} for key in keys}
    seds_duration_dict['all'] = {}
    seds_ground_truth_dict = {key: {} for key in keys}
    seds_ground_truth_dict['all'] = {}
    dataset_list = []

    for key in keys:
        cur_duration_path = os.path.join(tsv_root, f'{key}_duration.tsv')
        cur_tsv_path = os.path.join(tsv_root, f'{key}.tsv')
        cur_audio_root = os.path.join(audio_root, key)

        cur_seds_durations = get_sed_duration(path=cur_duration_path)
        cur_seds_ground_truth = get_sed_ground_truth(path=cur_tsv_path,
                                                     audio_root=cur_audio_root,
                                                     test_positive_labels=test_positive_labels)

        seds_duration_dict[key] = cur_seds_durations
        for k, v in cur_seds_durations.items():
            assert k not in seds_duration_dict['all'].keys()
            seds_duration_dict['all'][k] = v
        seds_ground_truth_dict[key] = cur_seds_ground_truth
        for k, v in cur_seds_ground_truth.items():
            assert k not in seds_ground_truth_dict['all'].keys()
            seds_ground_truth_dict['all'][k] = v
    return seds_duration_dict, seds_ground_truth_dict


def get_result(test_results, seds_ground_truth0, seds_durations0, frames_per_second):
    seds_result = get_seds_pred(test_results=test_results,
                                frames_per_second=frames_per_second,
                                seds_ground_truth0=seds_ground_truth0,
                                seds_durations0=seds_durations0)
    work_thrs = np.arange(0.05, 1., 0.05).tolist()
    for key1 in seds_result.keys():
        for key2 in seds_result[key1].keys():
            if key2.find('threshold') < 0: continue
            if isinstance(seds_result[key1][key2], list):
                work_thrs.extend(seds_result[key1][key2])
            else:
                work_thrs.append(seds_result[key1][key2])

    work_thrs = list(set(work_thrs))
    work_thrs.sort()
    other_result = []
    for thr in work_thrs:
        other_result.append(get_other_metrics(thr=thr,
                                              pred=test_results,
                                              ground_truth=seds_ground_truth0,
                                              valid_durations=seds_durations0,
                                              frames_per_second=frames_per_second))
    return seds_result, other_result



"""
conda activate atst_sed_dasheng
cd /home/xingyuan/animal_language2/code/ATST-SED2/beats_work

python test.py


"""


if __name__ == '__main__':
    config_path = '/data/xingyuan/animal_language/v2/out/second_cut_model/beats/stage1/version_1/hparams.yaml'
    checkpoint_path = '/data/xingyuan/animal_language/v2/out/second_cut_model/beats/stage1/version_1/checkpoints/epoch=109-step=6160.ckpt'
    thr = 0.6712402105331421
    model_mode = 'student'
    device = torch.device('cuda:0')

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
        mode=model_mode
    )
    model.eval()
    model = model.to(device)

    frames_per_second = config["data"]["fs"] / (config["data"]["net_subsample"] * config["feats"]["hop_length"])
    seds_duration_dict, seds_ground_truth_dict = get_test_dataset()
    input_audio_paths = []
    audio_root = '/home/xingyuan/animal_language2/data/dog_sed_dataset2/strong_valid_audios_16k'
    for k in ['chihuahua', 'husky', 'shiba_inu', 'pitbull', 'labrador', 'german_shepherd']:
        r = os.path.join(audio_root, k)
        for f in os.listdir(r):
            input_audio_paths.append(os.path.join(r, f))
    min_length = int(10. * 16000)
    pred = {}
    for path in tqdm(input_audio_paths):
        mixture, onset_s, offset_s, padded_indx = read_audio(path, False, False, None)

        # mixture, _ = librosa.load(path, sr=16000, mono=True)
        # mixture = torch.tensor(mixture)
        # mixture = mixture.to(device)
        if mixture.shape[0] < min_length:
            mixture = torch.nn.functional.pad(mixture,
                                              (0, min_length - mixture.shape[0]),
                                              mode='constant',
                                              value=0.)
            assert mixture.shape[0] == min_length
        mixture = mixture.to(device)

        with torch.no_grad():
            model.eval()
            sed_results = inference_model(mixture=mixture, model=model)

        key = os.path.basename(path)
        pred[key] = sed_results

    seds_result, other_result = get_result(pred, seds_ground_truth_dict['all'], seds_duration_dict['all'], frames_per_second)

    print(seds_result)
    print(other_result)
