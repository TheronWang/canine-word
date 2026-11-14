import os
import json
import shutil
import numpy as np


def merge_consecutive_transcriptions(transcription) -> list[int]:
    """
    Merge consecutive frames with the same transcription.

    Args:
        transcription: list or ndarrays of transcriptions.
        
    Example:
        >>> transcription = [1, 1, 2, 2, 3, 3, 1, 1]
        >>> result = [1, 2, 3, 1]
    """
    # handle empty transcription
    if len(transcription) == 0:
        return []
    
    # handle numpy array; transform to list
    if isinstance(transcription, np.ndarray):
        transcription = transcription.tolist()
    
    result = [transcription[0]]
    for i in range(1, len(transcription)):
        if transcription[i] != transcription[i - 1]:
            result.append(transcription[i])
    return [int(i) for i in result] 


def make_new_samples():
    sample_folder = "/home/sinong/ALP/AniLang/PhonemeFinding/checkpoints/phoneme_refined/phone_samples"
    mapping_file = "/home/sinong/ALP/AniLang/PhonemeFinding/checkpoints/phoneme_refined/241013_001019/mapping_phoneme.json"
    config_file = "/home/sinong/ALP/AniLang/PhonemeFinding/checkpoints/phoneme_refined/241013_001019/config.json"
    with open(mapping_file, "r") as f:
        mapping = json.load(f)
    with open(config_file, "r") as f:
        config = json.load(f)
    found_phonemes = config["found_phonemes"]
    sample_dict = {}
    for phone_name in os.listdir(sample_folder):
        phone_folder = os.path.join(sample_folder, phone_name)
        sample_dict[phone_name] = []
        for sample_name in os.listdir(phone_folder):
            sample_path = os.path.join(phone_folder, sample_name)
            sample_dict[phone_name].append(sample_path)
    for ori, rep in mapping.items():
        sample_dict[rep] += sample_dict[ori]
        del sample_dict[ori]
    for phone_name, sample_list in sample_dict.items():
        print(f"{phone_name}: {len(sample_list)}")
        
    new_sample_folder = "/home/sinong/ALP/AniLang/PhonemeFinding/checkpoints/phoneme_refined/hubert250-t5-p5-samples"
    os.makedirs(new_sample_folder, exist_ok=True)
    for phone_name, sample_list in sample_dict.items():
        if phone_name not in found_phonemes:
            continue
        phone_folder = os.path.join(new_sample_folder, phone_name)
        os.makedirs(phone_folder, exist_ok=True)
        for sample_path in sample_list:
            sample_name = os.path.basename(sample_path)
            new_sample_path = os.path.join(phone_folder, sample_name)
            shutil.copy(sample_path, new_sample_path)
 
        
if __name__ == "__main__":
    # transcription = [1, 1, 2, 2, 3, 3, 1, 1]
    # result = merge_consecutive_frames(transcription)
    make_new_samples()
