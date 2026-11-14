import json
import os.path
import shutil

import librosa
from pydub import AudioSegment
from tqdm import tqdm


def audio_filter(duration_data, output_root, min_duration=0.15, window_size=5., stride=4.):
    path = duration_data['path']
    assert path.find('_16k/') >= 0
    duration = duration_data['duration']

    if duration < min_duration:
        return True

    name = os.path.basename(path)[:-len('.wav')]
    if min_duration <= duration <= window_size:
        dst = os.path.join(output_root, f'{name}+0.0.wav')
        assert not os.path.exists(dst), dst
        shutil.copy(src=path, dst=dst)
        return True

    if duration > window_size:
        audio = AudioSegment.from_wav(path)
        start = 0.
        while start < duration:
            end = min(start + window_size, duration)
            if (end - start) < min_duration:
                start += stride
                continue
            cur_audio = audio[start * 1000.: end * 1000.]
            dst = os.path.join(output_root, f'{name}+{str(start)}.wav')
            assert not os.path.exists(dst), dst
            cur_audio.export(dst, format='wav')
            start += stride
        return True

    return False


if __name__ == '__main__':
    keys = ['chihuahua', 'husky', 'shiba_inu', 'pitbull', 'labrador', 'german_shepherd']
    root = '/data/xingyuan/animal_language/v2/out/second_cut/beats-sed_109-6160_6713'
    min_duration = 0.15
    window_size = 5.
    stride = 4.
    output_root = f'/data/xingyuan/animal_language/v2/out/fairseq/six_balance/data/{int(min_duration * 1000)}-{int(window_size * 1000)}-{int(stride * 1000)}'

    # if os.path.exists(output_root): shutil.rmtree(output_root)
    # os.makedirs(output_root)
    # print(output_root)
    # for key in keys:
    #     print(key)
    #     audio_root = os.path.join(root, f'{key}_thr0.05_offset1_0.5_audios_16k')
    #     duration_path = os.path.join(root, f'{key}_thr0.05_offset1_0.5_audios_16k_durations.jsonlines')
    #     with open(duration_path, 'r', encoding='utf-8') as file:
    #         duration_datas = [json.loads(line.strip()) for line in file.readlines() if len(line.strip()) > 0]
    #     cur_output_root = os.path.join(output_root, key)
    #     if os.path.exists(cur_output_root): shutil.rmtree(cur_output_root)
    #     os.makedirs(cur_output_root)
    #     for dd in tqdm(duration_datas):
    #         flag = audio_filter(duration_data=dd,
    #                             output_root=cur_output_root,
    #                             min_duration=min_duration,
    #                             window_size=window_size,
    #                             stride=stride)
    #         assert flag

    # # 合并所有音频
    # all_root = os.path.join(output_root, 'all')
    # os.makedirs(all_root, exist_ok=True)
    # for key in keys:
    #     cur_output_root = os.path.join(output_root, key)
    #     for file in tqdm(os.listdir(cur_output_root)):
    #         src = os.path.join(cur_output_root, file)
    #         dst = os.path.join(all_root, file)
    #         shutil.copy(src=src, dst=dst)


