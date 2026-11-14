import bisect
import json
import logging
import os
import sys

import fairseq
import librosa
import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from tqdm import tqdm

from feature_utils import get_path_iterator, dump_feature
from fairseq.data.audio.audio_utils import get_features_or_waveform
import logging
import os
import sys

import numpy as np

import joblib
import torch


class HubertFeatureReader(object):
    def __init__(self, ckpt_path, layer, max_chunk=160000000):
        (
            model,
            cfg,
            task,
        ) = fairseq.checkpoint_utils.load_model_ensemble_and_task([ckpt_path])
        self.model = model[0].eval().cuda()
        self.task = task
        self.layer = layer
        self.max_chunk = max_chunk

    def read_audio(self, path, ref_len=None):
        # wav = get_features_or_waveform(path, need_waveform=True, use_sample_rate=self.task.cfg.sample_rate)
        wav, _ = librosa.load(path, sr=self.task.cfg.sample_rate)
        if wav.ndim == 2:
            wav = wav.mean(-1)
        assert wav.ndim == 1, wav.ndim
        if ref_len is not None and abs(ref_len - len(wav)) > 160:
            logging.warning(f"ref {ref_len} != read {len(wav)} ({path})")
        return wav

    def get_feats(self, path, ref_len=None):
        x = self.read_audio(path, ref_len=ref_len)
        with torch.no_grad():
            x = torch.from_numpy(x).float().cuda()
            if self.task.cfg.normalize:
                x = F.layer_norm(x, x.shape)
            x = x.view(1, -1)

            feat = []
            for start in range(0, x.size(1), self.max_chunk):
                x_chunk = x[:, start : start + self.max_chunk]
                feat_chunk, _ = self.model.extract_features(
                    source=x_chunk,
                    padding_mask=None,
                    mask=False,
                    output_layer=self.layer,
                )
                feat.append(feat_chunk)
        return torch.cat(feat, 1).squeeze(0)


class ApplyKmeans(object):
    def __init__(self, km_path):
        self.km_model = joblib.load(km_path)
        self.C_np = self.km_model.cluster_centers_.transpose()
        self.Cnorm_np = (self.C_np ** 2).sum(0, keepdims=True)

        self.C = torch.from_numpy(self.C_np)
        self.Cnorm = torch.from_numpy(self.Cnorm_np)
        if torch.cuda.is_available():
            self.C = self.C.cuda()
            self.Cnorm = self.Cnorm.cuda()

    def __call__(self, x):
        if isinstance(x, torch.Tensor):
            dist = (
                x.pow(2).sum(1, keepdim=True)
                - 2 * torch.matmul(x, self.C)
                + self.Cnorm
            )
            return dist.argmin(dim=1).cpu().numpy()
        else:
            dist = (
                (x ** 2).sum(1, keepdims=True)
                - 2 * np.matmul(x, self.C_np)
                + self.Cnorm_np
            )
            return np.argmin(dist, axis=1)


def get_input_files():
    keys = ['chihuahua', 'husky', 'shiba_inu', 'pitbull', 'labrador', 'german_shepherd']
    root = '/data/xingyuan/animal_language/v2/out/second_cut/beats-sed_109-6160_6713'
    path2key = {}
    for key in tqdm(keys):
        cur_root = os.path.join(root, f'{key}_thr0.05_offset1_0.5_audios_16k')
        for audio_file in os.listdir(cur_root):
            audio_path = os.path.join(cur_root, audio_file)
            if audio_path in path2key.keys():
                path2key[audio_path].append(key)
            else:
                path2key[audio_path] = [key]

    return path2key


if __name__ == '__main__':
    output_root = '/data/xingyuan/animal_language/v2/out/fairseq/six_balance/work/iter2_130k_11th_km250'
    ckpt_path = '/data/xingyuan/animal_language/v2/out/fairseq/six_balance/iter2/res1/checkpoints/checkpoint_105_130000.pt'
    km_path = '/data/xingyuan/animal_language/v2/out/fairseq/six_balance/iter3/130k_11th_km250.pkl'
    layer = 11


    output_feature_root = os.path.join(output_root, 'feature')
    output_label_root = os.path.join(output_root, 'label')
    metadata_path = os.path.join(output_root, 'metadata.jsonlines')
    os.makedirs(output_feature_root, exist_ok=True)
    os.makedirs(output_label_root, exist_ok=True)
    path2key = get_input_files()

    input_datas = []
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r', encoding='utf-8') as file:
            done_datas = [json.loads(line.strip()) for line in file.readlines() if len(line.strip()) > 0]
        done_paths = [dd['audio_path'] for dd in done_datas]
        done_paths.sort()
        for path in path2key.keys():
            ind = bisect.bisect_left(done_paths, path)
            if (0 <= ind < len(done_paths)) and (done_paths[ind] == path):
                continue
            else:
                input_datas.append({
                    'audio_path': path,
                    'breeds': path2key[path]
                })
    else:
        for path in path2key.keys():
            input_datas.append({
                'audio_path': path,
                'breeds': path2key[path]
            })

    reader = HubertFeatureReader(ckpt_path, layer)
    apply_kmeans = ApplyKmeans(km_path)
    for i_data in tqdm(input_datas):
        name = os.path.basename(i_data['audio_path'])[:-len('.wav')]
        feat = reader.get_feats(i_data['audio_path'])
        label = apply_kmeans(feat)
        feat = feat.cpu().numpy()

        feat_path = os.path.join(output_feature_root, f'{name}.npy')
        label_path = os.path.join(output_label_root, f'{name}.npy')
        wav, sr = librosa.load(i_data['audio_path'], sr=None, mono=True)
        np.save(feat_path, feat)
        np.save(label_path, label)

        i_data['feature_path'] = feat_path
        i_data['label_path'] = label_path
        i_data['duration'] = wav.shape[0] / sr
        i_data['name'] = name

        with open(metadata_path, 'a+', encoding='utf-8') as file:
            file.write(json.dumps(i_data, ensure_ascii=False))
            file.write('\n')
            file.flush()


