import numpy as np
import json
import librosa
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from sklearn.metrics.pairwise import cosine_similarity


def extract_acoustic_feature(audio_file_path: str,
                             feature_type: str='mel',
                             average: bool=True) -> np.ndarray:
    y, sr = librosa.load(audio_file_path)
    feature = None
    if feature_type == 'mel':
        feature = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, n_fft=512, hop_length=64)
    elif feature_type == 'mfcc':
        feature = librosa.feature.mfcc(y=y, sr=sr)
    elif feature_type == 'stft':
        feature = librosa.stft(y=y, n_fft=512, hop_length=64)
        # convert complex number to two real numbers
        feature = np.stack((np.real(feature), np.imag(feature)), axis=-1)
    else:
        raise ValueError("Feature type not supported.")
    
    if average:
        feature = np.mean(feature, axis=1)

    # ======
    # Fixme: using filter bank instead; use mel spectrogram 128 dimension
    # 40ms window, 20ms hop length
    # ======
    return feature


def calculate_euclidean_distance(audio_path1: str,
                                 audio_path2: str,
                                 feature_type: str='mfcc') -> float:
    feature1 = extract_acoustic_feature(audio_path1,
                                             feature_type).reshape(1, -1)
    feature2 = extract_acoustic_feature(audio_path2,
                                             feature_type).reshape(1, -1)
    return float(np.linalg.norm(feature1 - feature2))


def calculate_cosine_similarity(audio_path1: str,
                                audio_path2: str,
                                feature_type: str='mfcc') -> float:
    feature1 = extract_acoustic_feature(audio_path1,
                                             feature_type).reshape(1, -1)
    feature2 = extract_acoustic_feature(audio_path2,
                                             feature_type).reshape(1, -1)
    
    distance = cosine_similarity(feature1, feature2)[0][0]
    return float(distance)


def calculate_dtw_distance(audio_path1: str,
                           audio_path2: str,
                           feature_type: str='mfcc') -> float:
    feature1 = extract_acoustic_feature(audio_path1,
                                            feature_type, average=False)
    feature2 = extract_acoustic_feature(audio_path2,
                                            feature_type, average=False)
    
    feature1, feature2 = feature1.T, feature2.T
    
    distance, path = fastdtw(feature1, feature2, dist=euclidean)
    
    return distance

    
def calculate_list_accuracy(predictions: list, ground_truths: list) -> float:
    # handle edge cases
    if len(predictions) == 0 or len(ground_truths) == 0:
        return 0.0
    # check the length and type of the input lists
    assert len(predictions) == len(ground_truths), \
        "The length of predictions and ground truths should be the same."
    assert type(predictions[0]) == type(ground_truths[0]), \
        "The type of predictions and ground truths should be the same."
    
    correct = 0
    for i in range(len(predictions)):
        if predictions[i] == ground_truths[i]:
            correct += 1
    return correct / len(predictions)


def regenerate_metadata_file(meta_file_path: str) -> None:
    with open(meta_file_path, "r") as f:
        metadata = json.load(f)
    md = metadata['metadata']
    answers = []
    
    for samples in md:
        answer = int(samples['answer'].split("_")[-1])
        answers.append(answer)
    
    metadata['answer'] = answers
    
    with open(meta_file_path, "w") as f:
        json.dump(metadata, f)


if __name__ == "__main__":
    regenerate_metadata_file("/home/sinong/ALP/DogRecognitionLabeling/grammar_induction/PyAdaGram/phone_evaluation_tri_pairs/triplet_pairs_phone_evaluation20240910174727/metadata.json")
