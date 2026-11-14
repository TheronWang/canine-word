import json
import os.path
import random
import shutil
import librosa
from tqdm import tqdm



if __name__ == '__main__':
    dog_names = [
        'chihuahua',
        'shiba_inu',
        'husky',
        'pitbull',
        'german_shepherd',
        'labrador',
    ]
    root = '/data/xingyuan/animal_language/v2/out/second_cut/beats-sed_109-6160_6713'
    for dog_name in dog_names:
        durations = []
        file_sizes = []
        audio_root = os.path.join(root, f'{dog_name}_thr0.05_offset1_0.5_audios_16k')
        for audio_file in tqdm(os.listdir(audio_root)):
            audio_path = os.path.join(audio_root, audio_file)
            wav, sr = librosa.load(audio_path, sr=None, mono=True)
            file_sizes.append(os.path.getsize(audio_path))
            durations.append(wav.shape[0] / sr)
        print()
        print(dog_name)
        print(f'size: {sum(file_sizes) >> 20}')
        print(f'count: {len(durations)}')
        print(f'sum: {sum(durations) / 3600.}')
        print(f'mean: {sum(durations) / len(durations)}')
        print(f'min: {min(durations)}')
        print(f'max: {max(durations)}')


