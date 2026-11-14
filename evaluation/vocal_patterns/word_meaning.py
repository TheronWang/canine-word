import os
import ast
import sys
import cv2
import json
import random
import shutil
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
from multiprocessing import Pool

sys.path.append("/home/sinong/ALP/AniLing/")
# from constant import cut_audio

import numpy as np
import librosa
import soundfile as sf
import opensmile


def cut_audio(
    audio_path: str, start_time: float, end_time: float, output_path: str
) -> None:
    """
    Cut the audio file from start_time to end_time and save it to output_path.

    Args:
    audio_path: str, path to the input audio file.
    start_time: float, start time in seconds.
    end_time: float, end time in seconds.
    output_path: str, path to save the output audio file.

    Returns:
    None
    """
    # Load the audio file
    audio, sr = librosa.load(audio_path, sr=None)

    # Convert start_time and end_time to samples
    start_sample = int(start_time * sr)
    end_sample = int(end_time * sr)

    # Cut the audio
    audio_cut = audio[start_sample:end_sample]

    # Save the audio
    sf.write(output_path, audio_cut, sr)
    return


def get_mfcc_fea(audio_path: str) -> np.ndarray:
    """
    Get the MFCC features of an audio file.

    Args:
    audio_path: str, path to the input audio file.

    Returns:
    mfcc: np.ndarray, MFCC features of the audio file.
    """
    # Load the audio file
    audio, sr = librosa.load(audio_path, sr=None)

    # Compute MFCC features
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    return mfcc


def get_mel_spectro_fea(audio_path: str) -> np.ndarray:
    """
    Get the Mel spectrogram features of an audio file.

    Args:
    audio_path: str, path to the input audio file.

    Returns:
    mel_spectro: np.ndarray, Mel spectrogram features of the audio file.
    """
    # Load the audio file
    audio, sr = librosa.load(audio_path, sr=None)

    # Compute Mel spectrogram features
    mel_spectro = librosa.feature.melspectrogram(y=audio, sr=sr)
    return mel_spectro


def get_gemaps_fea(audio_path: str) -> np.ndarray:
    """
    Get the GeMAPS features of an audio file.

    Args:
    audio_path: str, path to the input audio file.

    Returns:
    gemaps: np.ndarray, GeMAPS features of the audio file.
    """
    smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.GeMAPSv01b,
        feature_level=opensmile.FeatureLevel.Functionals,
    )
    fea = smile.process_file(audio_path)
    return fea


def get_egemaps_fea(audio_path: str) -> np.ndarray:
    """
    Get the eGeMAPS features of an audio file.

    Args:
    audio_path: str, path to the input audio file.

    Returns:
    egemaps: np.ndarray, eGeMAPS features of the audio file.
    """
    smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02,
        feature_level=opensmile.FeatureLevel.Functionals,
    )
    fea = smile.process_file(audio_path)
    return fea


def get_compare_fea(audio_path: str) -> np.ndarray:
    """
    Get the comparison features of an audio file.

    Args:
    audio_path: str, path to the input audio file.

    Returns:
    compare_fea: np.ndarray, comparison features of the audio file.
    """
    smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.ComParE_2016,
        feature_level=opensmile.FeatureLevel.Functionals,
    )
    fea = smile.process_file(audio_path)
    return fea


# transcription_meta_path = (
#     "/data/xingyuan/animal_language/v2/out/fairseq/"
#     "six_balance/work/iter2_130k_11th_km250/metadata_trans.json"
# )
# clips_meta_path = "/home/sinong/ALP/data/sentence_meta.json"
# raw_video_dict_path = "/home/sinong/ALP/AniLing/raw_video_path.json"
# with open(clips_meta_path) as f:
#     clips_meta = json.load(f)

# with open(transcription_meta_path) as f:
#     transcription_meta = json.load(f)

# with open(raw_video_dict_path) as f:
#     raw_video_dict = json.load(f)


# # Select a subset of words to process (top 100 most frequent words)
# phoneme_finding_config_path = (
#     "/home/sinong/ALP/AniLing/PhonemeFinding/checkpoints/"
#     "phoneme_refined/241013_180004/config.json"
# )
# with open(phoneme_finding_config_path, "r") as f:
#     phoneme_finding_config = json.load(f)
# word_dict = phoneme_finding_config["found_words"]
# # sort the words by the number of appearances and get top 100
# word_dict = dict(
#     sorted(word_dict.items(), key=lambda item: item[1], reverse=True)
# )
# WORD_LIST = dict(list(word_dict.items())[:100]).keys()


def get_raw_video_path():
    parent_dirs = ["/data/dog_videos", "/bigdrive/shiba"]
    raw_video_dict = {}

    # Go through all files under the parent directory no matter how deep
    # key should be the basename of the video file
    # value should be the full path to the video file
    for parent_dir in parent_dirs:
        for root, dirs, files in os.walk(parent_dir):
            for file in files:
                if file.endswith(".mp4") and not file.startswith("audio"):
                    video_id = os.path.basename(file)[: -len(".mp4")]
                    raw_video_dict[video_id] = os.path.join(root, file)

    # Save the raw video path
    with open("raw_video_path.json", "w") as f:
        json.dump(raw_video_dict, f, indent=4)
    return raw_video_dict


def replace_phones(word: str, replacement: dict):
    """
    Replace the phones in a word with the given replacement.
    """
    word_phone_list = word.split(" ")
    new_phone_list = []

    for phone in word_phone_list:
        if phone in replacement:
            new_phone = replacement[phone]
        else:
            new_phone = phone

        if new_phone_list:
            if new_phone_list[-1] == new_phone:
                continue
            else:
                new_phone_list.append(new_phone)
        else:
            new_phone_list.append(new_phone)

    new_word = " ".join(new_phone_list)
    return new_word


def get_frames_from_video(
    video_path: str,
    start_ts: float,
    end_ts: float,
    num_frames: int,
    output_path: str,
) -> list[str]:
    """
    Extract `num_frames` randomly selected frames between start_ts
    and end_ts from the video file, and save them in the specified
    output folder.

    Returns a list of the paths to the saved frame images.

    The output file names will be of the format:
    {video_filename_without_extension}_{int(start_ts)}
    -{int(end_ts)}_{frame_idx}.jpg
    """

    # Create output directory if it doesn't exist
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Get the base video filename (without directory and extension)
    video_filename = os.path.basename(video_path)
    video_name, _ = os.path.splitext(video_filename)

    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Error opening video file: {video_path}")

    # Get video properties: FPS and total frame count
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_duration = total_frames / fps

    # extend the start ts and end ts 3 seconds
    start_ts = max(0, start_ts - 3)
    end_ts = min(video_duration, end_ts + 3)

    # Ensure the specified timestamps are within the video duration
    if start_ts < 0 or start_ts > video_duration:
        raise ValueError("start_ts must be between 0 and the video duration.")
    if end_ts < 0 or end_ts > video_duration:
        end_ts = video_duration
    if start_ts >= end_ts:
        raise ValueError("start_ts must be less than end_ts.")

    # Randomly choose `num_frames` timestamps between start_ts and end_ts
    timestamps = [random.uniform(start_ts, end_ts) for _ in range(num_frames)]
    # Sorting timestamps can make frame seeking more efficient (optional)
    timestamps.sort()

    frames_img_paths = []

    for idx, ts in enumerate(timestamps, start=1):
        # Convert timestamp to frame index
        frame_index = int(ts * fps)
        # Set the video capture to the desired frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = cap.read()
        if ret:
            # Construct the output filename
            output_filename = (
                f"{video_name}_{int(start_ts)}-{int(end_ts)}_{idx}.jpg"
            )
            output_filepath = os.path.join(output_path, output_filename)
            # Save the frame as a JPEG file
            cv2.imwrite(output_filepath, frame)
            frames_img_paths.append(output_filepath)
            print(
                f"Saved frame at {ts:.2f}s (frame {frame_index}) to"
                f" {output_filepath}"
            )
        else:
            print(
                f"Warning: Could not retrieve frame at {ts:.2f}s (frame"
                f" {frame_index})."
            )

    # Release the video capture object
    cap.release()
    return frames_img_paths


def process_sentence(args):
    """
    Process a single sentence from transcription_meta.

    For the given sentence, this function:
      - Retrieves the video path using raw_video_dict.
      - Checks for required clips meta.
      - Extracts word meta by iterating over the parsed transcription and word intervals.
      - Applies phone replacement (if provided).
      - Retrieves image frames for the word segment.

    Returns a dictionary mapping each word to its metadata for this sentence.
    """
    sentence_id, sentence_meta, replacement = args
    local_words_meta = {}

    # Get the video path of this sentence
    try:
        video_id = sentence_id.split("+")[1]
        row_video_path = raw_video_dict[video_id]
    except KeyError:
        print(f"Video not found for sentence {sentence_id}")
        return local_words_meta  # Return empty dict if video not found

    if sentence_id not in clips_meta:
        print(f"Clips meta not found for sentence {sentence_id}")
        return local_words_meta

    # Extract the word meta (timestamps) for this sentence
    sentence_start = clips_meta[sentence_id]["start"]
    sentence_end = clips_meta[sentence_id]["end"]

    try:
        for clip_phones, clip_internals in zip(
            sentence_meta["parsed_transcription"],
            sentence_meta["word_intervals"],
        ):
            if clip_phones.startswith("w"):
                # Remove the leading "w" to get the actual word
                word = clip_phones[1:]

                # Check if any phone is replaced using the replacement mapping
                if replacement:
                    word = replace_phones(word, replacement)

                # Calculate word start and end timestamps
                word_start = sentence_start + clip_internals[0]
                word_end = sentence_start + clip_internals[1]

                # Extract frames for the word clip from the video
                word_frames_paths = get_frames_from_video(
                    row_video_path,
                    word_start,
                    word_end,
                    3,
                    "/data/_assets/dog_frames",
                )

                # Initialize or update the word meta for this sentence
                if word in local_words_meta:
                    if sentence_id in local_words_meta[word]:
                        local_words_meta[word][sentence_id][
                            "img_sample_path"
                        ].extend(word_frames_paths)
                    else:
                        local_words_meta[word][sentence_id] = {
                            "video_path": row_video_path,
                            "img_sample_path": word_frames_paths,
                            "environment": [],
                            "activity": [],
                            "intension": [],
                        }
                else:
                    local_words_meta[word] = {
                        sentence_id: {
                            "video_path": row_video_path,
                            "img_sample_path": word_frames_paths,
                            "environment": [],
                            "activity": [],
                            "intension": [],
                        }
                    }
    except Exception as e:
        print(f"Error processing sentence {sentence_id}: {e}")
    return local_words_meta


def merge_words_meta(dict_list):
    """
    Merge a list of words_meta dictionaries (one per sentence) into a single dictionary.

    The merged dictionary has the following structure:

    {
       "word_0": {
          "sentence_id_0": {
              "video_path": "/path/to/video",
              "img_sample_path": ["/path/to/img", ...],
              "environment": [...],
              "activity": [...],
              "intension": [...],
          },
          "sentence_id_1": { ... },
       },
       "word_1": { ... },
       ...
    }
    """
    merged = {}
    for local_dict in dict_list:
        for word, sentence_data in local_dict.items():
            if word not in merged:
                merged[word] = sentence_data
            else:
                for sentence_id, meta in sentence_data.items():
                    if sentence_id in merged[word]:
                        merged[word][sentence_id]["img_sample_path"].extend(
                            meta["img_sample_path"]
                        )
                    else:
                        merged[word][sentence_id] = meta
    return merged


def get_words_meta(replacement_file=None):
    """
    Generate the words_meta dictionary by processing each sentence in transcription_meta using multiprocessing.

    The words_meta dictionary structure is:

    {
       "word_0": {
          "sentence_id_0": {
              "video_path": "/path/to/video",
              "img_sample_path": ["/path/to/img", ...],
              "environment": [...],
              "activity": [...],
              "intension": [...],
          },
          "sentence_id_1": { ... },
       },
       "word_1": { ... },
       ...
    }

    If a replacement_file is provided, it is used to load phone replacement mappings.
    """
    # Load the replacement file if provided, else use an empty dictionary.
    if replacement_file is not None:
        with open(replacement_file) as f:
            replacement = json.load(f)
    else:
        replacement = {}

    # Initialize the words_meta dictionary (will be merged from all worker processes)
    words_meta = {}

    # Build a list of tasks: each task is (sentence_id, sentence_meta, replacement)
    tasks = [
        (sentence_id, sentence_meta, replacement)
        for sentence_id, sentence_meta in transcription_meta.items()
    ]

    # Use a multiprocessing Pool to process sentences in parallel.
    with Pool() as pool:
        # Each worker processes a sentence and returns its local words_meta dictionary.
        local_results = pool.map(process_sentence, tasks)

    # Merge the local dictionaries from all workers into one words_meta dictionary.
    words_meta = merge_words_meta(local_results)

    # Save the words_meta to a JSON file.
    with open("words_meta.json", "w") as f:
        json.dump(words_meta, f, indent=4)

    return words_meta


def analysis_word_pool_meaning(word_meaning_file_path: str):
    with open(word_meaning_file_path) as f:
        word_meaning = json.load(f)
    environment_words_pool = defaultdict(int)
    activity_words_pool = defaultdict(int)
    intension_words_pool = defaultdict(int)

    for word, sentence_data in word_meaning.items():
        print(f"Addressing Word: {word}")
        for sentence_id, meta in sentence_data.items():
            for key in ["environment", "activity", "intension"]:
                for word_list in meta[key]:
                    if key == "environment":
                        for word in word_list:
                            environment_words_pool[word] += 1
                    elif key == "activity":
                        for word in word_list:
                            activity_words_pool[word] += 1
                    else:
                        for word in word_list:
                            intension_words_pool[word] += 1
    environment_words_pool = dict(
        sorted(
            environment_words_pool.items(),
            key=lambda item: item[1],
            reverse=True,
        )
    )
    activity_words_pool = dict(
        sorted(
            activity_words_pool.items(),
            key=lambda item: item[1],
            reverse=True,
        )
    )
    intension_words_pool = dict(
        sorted(
            intension_words_pool.items(),
            key=lambda item: item[1],
            reverse=True,
        )
    )

    word_meaning_pool = {
        "environment": environment_words_pool,
        "activity": activity_words_pool,
        "intension": intension_words_pool,
    }
    with open("word_meaning_pool.json", "w") as f:
        json.dump(word_meaning_pool, f, indent=4)
    print("Word meaning pool saved.")


def analysis_word_meaning(word_meaning_file_path: str):
    with open(word_meaning_file_path) as f:
        word_meaning_list = json.load(f)
    word_meaning = dict()

    for word, sentence_data in word_meaning_list.items():
        this_word_meaning = {
            "environment": defaultdict(int),
            "activity": defaultdict(int),
            "intension": defaultdict(int),
        }
        for sentence_id, meta in sentence_data.items():
            for key in ["environment", "activity", "intension"]:
                for description_list in meta[key]:
                    for description in description_list:
                        this_word_meaning[key][description] += 1

        for key in ["environment", "activity", "intension"]:
            this_word_meaning[key] = dict(
                sorted(
                    this_word_meaning[key].items(),
                    key=lambda item: item[1],
                    reverse=True,
                )
            )
        word_meaning[word] = this_word_meaning

    with open("categorized_word_meaning.json", "w") as f:
        json.dump(word_meaning, f, indent=4)


# The 10 environment and 10 activity categories of the paper (Table 8),
# as named in category_mapping/*.csv.
ENVIRONMENT_CATEGORIES = [
    "Indoor Residential Spaces", "Walls, Windows", "Furniture & Household Items",
    "Outdoor General Spaces", "Plants & Greenery", "Play & Recreation Area",
    "Vehicles, Transportation Area", "Water & Wet Environment",
    "Pet-Related Space", "Human Presence & Crowd",
]
ACTIVITY_CATEGORIES = [
    "Passive Actions", "Movement & Locomotion", "Observing & Watching",
    "Exploring & Investigating", "Interaction with Items", "Feeding", "Drinking",
    "Interaction with Animals", "Water-Related Activities", "Interaction with Humans",
]
CATEGORIES = {"environment": ENVIRONMENT_CATEGORIES, "activity": ACTIVITY_CATEGORIES}
MAPPING_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "category_mapping")


def merge_word_meaning(word_meaning_file_path: str):
    """Map each Janus-Pro description to one of the 10 environment / 10 activity
    categories. Descriptions that map to no category are dropped."""
    with open(word_meaning_file_path) as f:
        word_meaning_list = json.load(f)
    mappings = {}
    for key, name in (("environment", "Environment"), ("activity", "Activity")):
        df = pd.read_csv(os.path.join(MAPPING_DIR, f"Final_Full_{name}_Mapping.csv"))
        mappings[key] = dict(zip(df["Original"], df["Merged Category"]))

    for word, sentence_data in word_meaning_list.items():
        for sentence_id, meta in sentence_data.items():
            for key in CATEGORIES:
                meta[key] = [
                    [mappings[key][d] for d in description_list
                     if mappings[key].get(d) in CATEGORIES[key]]
                    for description_list in meta.get(key, [])
                ]
    with open("categorized_selected_word_meta.json", "w") as f:
        json.dump(word_meaning_list, f, indent=4)


def plot_linechart(
    data: dict,
    output_path: str,
    title: str = None,
    upper_boundary: float = None,
    lower_boundary: float = None,
):
    """
    Plots a line chart from a dictionary and saves it as a PNG file with vertical x-axis labels.

    Parameters:
    - data (dict): Dictionary with keys as x-axis labels and values as y-axis float values.
    - output_path (str): Path to save the output PNG file.
    """
    # Extract x and y values
    x_values = list(data.keys())
    y_values = list(data.values())

    # Create the plot
    plt.figure(figsize=(10, 5))  # Increase figure size for better readability
    plt.plot(
        x_values,
        y_values,
        marker="o",
        linestyle="-",
        color="b",
        label="Relative Frequency",
    )

    # Customize the chart
    # plt.xlabel("X-axis")
    # plt.ylabel("Y-axis")
    if title:
        plt.title(title)
    else:
        plt.title("Line Chart with Vertical X-axis Labels")
    plt.xticks(rotation=45, fontsize=14)  # Rotate x-axis labels vertically
    if upper_boundary is not None and lower_boundary is not None:
        plt.ylim(lower_boundary, upper_boundary)
    plt.legend()
    plt.grid(True)

    # Save the figure
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()  # Close the figure to free memory


def plot_word_distribution():
    file_path = "/home/sinong/ALP/AniLing/word_statistics.txt"
    with open(file_path, "r") as f:
        content = f.read().strip()

    data_list = ast.literal_eval(content)
    data = dict(data_list)
    data = dict(sorted(data.items(), key=lambda item: item[1], reverse=True))
    points = list(data.values())[:200]

    # Create a new figure with a specified size (optional)
    plt.figure(figsize=(8, 6))

    # Plot the data as a curve with markers at each data point.
    plt.plot(points, marker="o", linestyle="-", color="b")

    # Label the axes and add a title
    plt.xlabel("Words")
    plt.ylabel("Counts")
    plt.title("Top 200 Words Distribution follows Zipf's Law")
    plt.xticks([])
    # Add a grid for better readability
    plt.grid(True)

    # Save the figure as a PNG file in the current directory
    plt.savefig("curve_plot.png")


def plot_word_distribution2():
    file_path = "/home/sinong/ALP/AniLing/word_statistics.txt"

    with open(file_path, "r") as f:
        content = f.read().strip()

    data_list = ast.literal_eval(content)
    data = dict(data_list)
    data = dict(sorted(data.items(), key=lambda item: item[1], reverse=True))
    points = list(data.values())[:200]

    # Create figure
    plt.figure(figsize=(8, 6))

    # Plot the actual word distribution as scatter points
    plt.scatter(
        range(1, len(points) + 1),
        points,
        marker="o",
        color="b",
        label="Word Frequency",
    )

    # Generate 1/x reference curve
    x_values = np.arange(1, len(points) + 1)  # Rank values (avoid zero)
    y_values = points[0] / x_values  # Scale 1/x by max frequency for reference

    # Plot the reference 1/x curve
    plt.plot(
        x_values, y_values, linestyle="--", color="red", label="1/x Reference"
    )

    # Label the axes and add a title
    plt.xlabel("Word Rank")
    plt.ylabel("Word Frequency")
    plt.title("Top 200 Words Distribution with Zipf's Law Reference")

    # Set logarithmic scale (optional, useful for Zipf's Law)
    plt.yscale("log")
    plt.xscale("log")

    # Add a legend
    plt.legend()

    # Add a grid
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)

    # Save the figure
    plt.savefig("curve_plot.png")


class wordMeaningAnalyzer:
    """
    This class is used to analyze the word meaning from the word meaning file.

    The words_meta dictionary structure is:
    {
       "word_0": {
          "sentence_id_0": {
              "video_path": "/path/to/video",
              "img_sample_path": ["/path/to/img", ...],
              "environment": [[], [], ...],
              "activity": [[], [], ...],
              "intension": [[], [], ...],
          },
          "sentence_id_1": { ... },
       },
       "word_1": { ... },
       ...
    }

    """

    def __init__(
        self,
        word_meta_file_path: str,
        word_description_mapping_file: str,
        phone_replacement_file_path: str,
        sentence_meta_file_path: str,
        transcription_meta_file_path: str,
    ):
        # set the file paths
        self.word_meta_file_path = word_meta_file_path
        self.word_description_mapping_file = word_description_mapping_file
        self.phone_replacement_file_path = phone_replacement_file_path
        self.sentence_meta_file_path = sentence_meta_file_path
        self.transcription_meta_file_path = transcription_meta_file_path

        # initialize the data
        self.word_meta = None
        self.word_description_mapping = None
        self.phone_replacement_mapping = None
        self.sentence_meta = None
        self.transcription_meta = None

    def _load_word_meta(self):
        if not self.sentence_meta_file_path:
            raise ValueError("sentence_meta_file_path is not set.")

        if not os.path.exists(self.sentence_meta_file_path):
            raise ValueError(
                "sentence_meta_file_path"
                f" {self.sentence_meta_file_path} does not exist."
            )

        with open(self.word_meta_file_path) as f:
            self.word_meta = json.load(f)

    def _load_word_description_mapping(self):
        if not self.word_description_mapping_file:
            raise ValueError("word_description_mapping_file is not set.")

        if not os.path.exists(self.word_description_mapping_file):
            raise ValueError(
                "word_description_mapping_file"
                f" {self.word_description_mapping_file} does not exist."
            )

        with open(self.word_description_mapping_file) as f:
            self.word_description_mapping = json.load(f)

    def _load_phone_replacement_mapping(self):
        if not self.phone_replacement_file_path:
            raise ValueError("phone_replacement_file_path is not set.")

        if not os.path.exists(self.phone_replacement_file_path):
            raise ValueError(
                "phone_replacement_file_path"
                f" {self.phone_replacement_file_path} does not exist."
            )

        with open(self.phone_replacement_file_path) as f:
            self.phone_replacement_mapping = json.load(f)

    def _load_sentence_meta(self):
        if not self.sentence_meta_file_path:
            raise ValueError("sentence_meta_file_path is not set.")

        if not os.path.exists(self.sentence_meta_file_path):
            raise ValueError(
                "sentence_meta_file_path"
                f" {self.sentence_meta_file_path} does not exist."
            )

        with open(self.sentence_meta_file_path) as f:
            self.sentence_meta = json.load(f)

    def _load_transcription_meta(self):
        if not self.transcription_meta_file_path:
            raise ValueError("transcription_meta_file_path is not set.")

        if not os.path.exists(self.transcription_meta_file_path):
            raise ValueError(
                "transcription_meta_file_path"
                f" {self.transcription_meta_file_path} does not exist."
            )

        with open(self.transcription_meta_file_path) as f:
            self.transcription_meta = json.load(f)

    def generate_word_audio_samples(
        self,
        output_dir: str,
        num_samples: int,
        extended_sample: bool = False,
    ):
        """
        Generate audio samples for each word in the word meta file.
        """
        # load required data if they are not loaded yet
        if not self.word_meta:
            self._load_word_meta()

        if not self.sentence_meta:
            self._load_sentence_meta()

        if not self.phone_replacement_mapping:
            self._load_phone_replacement_mapping()

        if not self.transcription_meta:
            self._load_transcription_meta()

        # create output directory if it does not exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # get candidate words list
        tobe_sampled_words = {word: 0 for word in self.word_meta.keys()}
        idx_word_mapping = {
            str(idx): word for idx, word in enumerate(tobe_sampled_words)
        }
        word_idx_mapping = {
            word: idx for idx, word in idx_word_mapping.items()
        }

        for sentence_id, trans_meta in self.transcription_meta.items():
            if "parsed_transcription" not in trans_meta:
                continue
            if "word_intervals" not in trans_meta:
                continue
            for word, interval in zip(
                trans_meta["parsed_transcription"],
                trans_meta["word_intervals"],
            ):
                if word.startswith("w"):
                    word = word[1:]
                    word = replace_phones(word, self.phone_replacement_mapping)

                    # if this word is the target word, then sample it
                    if word in tobe_sampled_words:
                        tobe_sampled_words[word] += 1

                        sentence_audio_path = (
                            "/data/shared/private/dogs/sentences/beats-sed_109-6160-6713"
                            + self.transcription_meta[sentence_id][
                                "audio_path"
                            ][72:]
                        )
                        start_ts = interval[0]
                        end_ts = interval[1]
                        word_audio_name = (
                            sentence_id
                            + str(start_ts)
                            + "_"
                            + str(end_ts)
                            + ".wav"
                        )
                        # create word folder if it does not exist
                        word_folder = os.path.join(
                            output_dir, "word_" + word_idx_mapping[word]
                        )
                        if not os.path.exists(word_folder):
                            os.makedirs(word_folder)
                            with open(
                                os.path.join(word_folder, "word.txt"),
                                "w",
                            ) as f:
                                f.write(word)

                        word_audio_path = os.path.join(
                            output_dir,
                            "word_" + word_idx_mapping[word],
                            word_audio_name,
                        )
                        cut_audio(
                            sentence_audio_path,
                            start_ts,
                            end_ts,
                            word_audio_path,
                        )

                        if extended_sample:
                            # extend the sample by 3 seconds
                            start_ts = max(0, start_ts - 0.5)
                            end_ts = min(
                                self.transcription_meta[sentence_id][
                                    "duration"
                                ],
                                end_ts + 0.5,
                            )
                            word_audio_name = (
                                "extend_"
                                + sentence_id
                                + str(start_ts)
                                + "_"
                                + str(end_ts)
                                + ".wav"
                            )
                            word_audio_path = os.path.join(
                                output_dir,
                                "word_" + word_idx_mapping[word],
                                word_audio_name,
                            )
                            cut_audio(
                                sentence_audio_path,
                                start_ts,
                                end_ts,
                                word_audio_path,
                            )
                        if tobe_sampled_words[word] >= num_samples:
                            del tobe_sampled_words[word]

                if len(tobe_sampled_words) == 0:
                    return

    def generate_word_meaning_distribution_plot(self, output_dir: str):
        """Relative frequency of each category for every vocal pattern (Eq. 1):

            RF(w, c) = C(c, w) / C(w)  -  sum_w' C(c, w') / sum_w' C(w')

        C(c, w) counts how often category c is described for occurrences of
        pattern w, and C(w) is the total count for w. Positive values mean the
        category is over-represented for w. One line chart per pattern and
        dimension is written to output_dir, and all values to
        word_frequency_distribution.json.
        """
        if not self.word_meta:
            self._load_word_meta()

        counts = {}  # counts[word][key][category] = C(c, w)
        for word, word_meta in self.word_meta.items():
            counts[word] = {key: {c: 0 for c in cats} for key, cats in CATEGORIES.items()}
            for sentence_meta in word_meta.values():
                for key in CATEGORIES:
                    for description_list in sentence_meta.get(key, []):
                        for category in description_list:
                            if category in counts[word][key]:
                                counts[word][key][category] += 1

        relative_frequency = {}
        for key, cats in CATEGORIES.items():
            global_total = sum(sum(counts[w][key].values()) for w in counts)
            global_freq = {
                c: (sum(counts[w][key][c] for w in counts) / global_total) if global_total else 0.0
                for c in cats
            }
            for word in counts:
                word_total = sum(counts[word][key].values())
                relative_frequency.setdefault(word, {})[key] = {
                    c: ((counts[word][key][c] / word_total) if word_total else 0.0) - global_freq[c]
                    for c in cats
                }

        for word, rf in relative_frequency.items():
            if len(word) > 100:
                continue
            for key in CATEGORIES:
                out_path = os.path.join(output_dir, "word_meaning_distribution", key,
                                        word.replace(" ", "_") + ".png")
                os.makedirs(os.path.dirname(out_path), exist_ok=True)
                plot_linechart(rf[key], out_path, title=(word + "    " + key))

        relative_frequency = {w: v for w, v in relative_frequency.items() if len(w) <= 100}
        with open("word_frequency_distribution.json", "w") as f:
            json.dump(relative_frequency, f, indent=4)


if __name__ == "__main__":

    replacement_file = (
        "/home/theron/ALP/AniLing/PhonemeFinding/checkpoints/"
        "phoneme_refined/241013_001019/mapping_phoneme.json"
    )
    transcription_meta_path = (
        "/data/data/data/xingyuan/animal_language/v2/out/fairseq/"
        "six_balance/work/iter2_130k_11th_km250/metadata_trans.json"
    )
    # word_meta_file_path = "/data/Janus/selected_word_meta.json"
    word_meta_file_path = (
        "/home/theron/ALP/AniLing/categorized_selected_word_meta.json"
    )
    word_description_mapping_file = ""
    phone_replacement_file_path = replacement_file
    sentence_meta_file_path = "/data/bigdrive/_assets/sentence_meta.json"
    transcription_meta_path = transcription_meta_path
    # get_raw_video_path()
    # get_words_meta(replacement_file)

    # # get top 100 words
    # word_counts_file = "/home/sinong/ALP/AniLing/word_counts.json"
    # with open(word_counts_file) as f:
    #     word_counts = json.load(f)
    # word_list = list(word_counts.keys())[:100]

    # # load words_meta data
    # words_meta_file = "/home/sinong/ALP/AniLing/words_meta.json"
    # with open(words_meta_file) as f:
    #     words_meta = json.load(f)

    # analysis_word_meaning(
    #     word_meaning_file_path="/home/sinong/ALP/AniLing/categorized_selected_word_meta.json"
    # )
    word_analyzer = wordMeaningAnalyzer(
        word_meta_file_path,
        word_description_mapping_file,
        phone_replacement_file_path,
        sentence_meta_file_path,
        transcription_meta_path,
    )
    word_analyzer.generate_word_audio_samples(
        output_dir=(
            "/home/theron/projects/web_apps/canine/data/pattern_samples"
        ),
        num_samples=10,
        extended_sample=True,
    )
    # word_analyzer.generate_word_meaning_distribution_plot("/data/_assets/")
    # plot_word_distribution()
    # merge_word_meaning(
    #     word_meaning_file_path="/data/Janus/selected_word_meta.json"
    # )

    # with open(sentence_meta_file_path, "r") as f:
    #     sentence_meta = json.load(f)
    # sentences = list(sentence_meta.keys())
    # # randomly pick 20 samples
    # selected_sentences = random.sample(sentences, 20)
    # # copy the selected sentences to a new folder
    # if not os.path.exists("/data/_assets/sentence_samples"):
    #     os.makedirs("/data/_assets/sentence_samples")
    # for sentence in selected_sentences:
    #     shutil.copyfile(
    #         sentence_meta[sentence]["path"],
    #         f"/data/_assets/sentence_samples/{sentence}.wav",
    #     )
    # plot_word_distribution2()
    m = 1
