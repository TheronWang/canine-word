import bisect
import json
import os.path
import random

import librosa
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


def show_raw_hist(durations, key):
    min_duration = 0.15
    max_duration = 5

    durations.sort()
    assert durations[0] >= min_duration
    assert durations[-1] <= max_duration
    for i in np.arange(0.35, 0.41, 0.01):
        ed = [d for d in durations if d < i]
        c_r = len(ed) / len(durations)
        d_r = sum(ed) / sum(durations)
        print(f'{i}: {c_r}\t{d_r}')
    plt.hist(durations, bins=50, edgecolor='black', range=(min(durations), max_duration), density=True)
    plt.axvline(x=min_duration, color='r', linestyle='--')
    plt.title(key)
    plt.savefig(f'{key}.png')
    plt.close()


def sample_data(min_duration, max_duration, key, total_count=-1, total_durations=-1.):
    root = '/data/xingyuan/animal_language/v2/out/second_cut/beats-sed_109-6160_6713'
    durations_path = os.path.join(root, f'{key}_thr0.05_offset1_0.5_audios_16k_durations.jsonlines')
    with open(durations_path, 'r', encoding='utf-8') as file:
        audios = [json.loads(line.strip()) for line in file.readlines() if len(line.strip()) > 0]
    audios.sort(key=lambda a: a['path'])
    random.seed(0)
    random.shuffle(audios)

    if total_count < 0:
        total_count = len(audios)
    if total_durations < 0:
        total_durations = 1000. * 60 * 60
    count = 0
    ef_durations = 0.
    ef_paths = []
    for audio in audios:
        if (count >= total_count) or (ef_durations >= total_durations): break
        if audio['duration'] < min_duration: continue
        count += 1
        ef_durations += min(audio['duration'], max_duration)
        ef_paths.append(audio['path'])
    return ef_paths, count, ef_durations


def show_cluster(path):
    with open(path, 'r', encoding='utf-8') as file:
        datas = [line.strip().split(',') for line in file.readlines() if len(line.strip()) > 0]
    for data in datas:
        for i in range(1, len(data)):
            data[i] = float(data[i])
    # datas.sort(key=lambda d: d[1])
    # plt.plot([d[1] for d in datas], [d[3] for d in datas])
    # plt.savefig('inertia.png')

    # datas.sort(key=lambda d: d[1])
    # plt.plot([d[1] for d in datas], [d[4] for d in datas])
    # plt.savefig('silhouette_score.png')

    datas.sort(key=lambda d: d[1])
    plt.plot([d[1] for d in datas], [d[5] for d in datas])
    plt.savefig('calinski_harabasz_score.png')



if __name__ == '__main__':
    keys = ['chihuahua', 'husky', 'shiba_inu', 'pitbull', 'labrador', 'german_shepherd']

    # path = '/data/xingyuan/animal_language/v2/out/fairseq/six_balance/iter3/find_cluster_from_iter2_130k_11th.txt'
    # show_cluster(path)

    path = '/data/xingyuan/animal_language/dog_data/labrador/metadata.jsonlines'
    with open(path, 'r', encoding='utf-8') as file:
        datas = [json.loads(line.strip()) for line in file.readlines() if len(line.strip()) > 0]
    print(len(datas))
    print(datas[0])

    # 找重复
    # # root = '/data/xingyuan/animal_language/v2/out/second_cut/beats-sed_109-6160_6713'
    # root = '/data/xingyuan/animal_language/dog_data'
    # key2audios = {key: [] for key in keys}
    # key2audios['all'] = []
    # for key in keys:
    #     # audio_root = os.path.join(root, f'{key}_thr0.05_offset1_0.5_audios_16k')
    #     audio_root = os.path.join(root, key, 'audios')
    #     key2audios[key] = [f for f in os.listdir(audio_root)]
    #     key2audios[key].sort()
    #     key2audios['all'].extend(key2audios[key])
    # key2audios['all'] = list(set(key2audios['all']))
    #
    # count = 0
    # for audio in tqdm(key2audios['all']):
    #     cur_keys = []
    #     for key in keys:
    #         ind = bisect.bisect_left(key2audios[key], audio)
    #         if (0 <= ind < len(key2audios[key])) and (key2audios[key][ind] == audio):
    #             cur_keys.append(key)
    #     assert len(cur_keys) >= 1
    #     if len(cur_keys) >= 2:
    #         count += 1
    #         print(audio)
    #         print(cur_keys)
    # print(count)



    # for key in keys + ['all']:
    #     root = '/data/xingyuan/animal_language/v2/out/fairseq/six_balance/data/150-5000-4000'
    #     output_path = os.path.join(root, f'{key}_durations.jsonlines')
    #     assert not os.path.exists(output_path)
    #     audio_root = os.path.join(root, key)
    #     files = [os.path.join(audio_root, f) for f in os.listdir(audio_root)]
    #     for f in tqdm(files):
    #         wav, sr = librosa.load(f, sr=None, mono=True)
    #         assert sr == 16000
    #         duration = {
    #             'path': f,
    #             'duration': wav.shape[0] / sr
    #         }
    #         with open(output_path, 'a+', encoding='utf-8') as file:
    #             file.write(json.dumps(duration, ensure_ascii=False))
    #             file.write('\n')
    #
    #     with open(output_path, 'r', encoding='utf-8') as file:
    #         datas = [json.loads(line.strip()) for line in file.readlines() if len(line.strip()) > 0]
    #     durations = [d['duration'] for d in datas]
    #     print(key)
    #     print(f'total duration: {sum(durations) / 3600.}')
    #     print(f'mean: {np.mean(durations)}')
    #     print(f'max: {max(durations)}')
    #     print(f'min: {min(durations)}')
    #     print(f'count: {len(durations)}')
    #     print()


    # path = '/home/xingyuan/animal_language2/data/dog_sed_dataset2/util.out'
    # with open(path, 'r', encoding='utf-8') as file:
    #     datas = [line.strip() for line in file.readlines() if len(line.strip()) > 0]
    # print(datas[-1])


    # path = '/data/xingyuan/animal_language/v2/out/fairseq/six_balance/data/150-5000-4000/all_durations.jsonlines'
    # with open(path, 'r', encoding='utf-8') as file:
    #     datas = [json.loads(line.strip()) for line in file.readlines() if len(line.strip()) > 0]
    # durations = [d['duration'] for d in datas]
    # show_raw_hist(durations, 'all')



