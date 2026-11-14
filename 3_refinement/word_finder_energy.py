import auditok
import os, math, json
import librosa
import numpy as np
import pandas as pd
import time
import argparse

import auditok.signal
from collections import defaultdict, Counter
from tqdm import tqdm

from utils import merge_consecutive_transcriptions


class WordFinder():
    def __init__(self, result_dir: str) -> None:
        self.result_dir = result_dir
        self.transcriptions_dir = os.path.join(result_dir, "label")
        self.metadata = dict()
        self.transcriptions = dict()
        
        # ========================================
        # Load the metadata and assign metadata to audio file names
        # Each metadata is a dictionary with the following keys:
        # "audio_path": str,  # .wav file
        # "breeds": list(str),
        # "feature_path": str, # .npy file
        # "label_path": str, # .npy file
        # "duration": float, # seconds
        # "name": str, # audio file name
        # ========================================
        metadata_file_path = os.path.join(result_dir, "metadata.jsonlines")
        with open(metadata_file_path, "r") as f:
            md = [json.loads(line) for line in f.readlines()]
            for data in md:
                self.metadata[data["name"]] = data
        
        self._load_transciptions()
        self._merge_transription_get_ts()
        
    def _load_transciptions(self) -> None:
        print("Loading transcriptions...")
        start_time = time.time()
        for audio_name, data in self.metadata.items():
            # load the raw transcription of the audio file
            raw_transcription = np.load(data["label_path"])
            
            # process the raw transcription
            # combine the joint frames with the same phoneme
            transcription = raw_transcription
            
            self.transcriptions[audio_name] = [int(phone) for phone in transcription]
        print(f"Transcriptions loaded. And used time: {time.time() - start_time:.2f} seconds.")
    
    def _parse_by_energy(self,
                         audio_path: str,
                         duration: float,
                         transcription: list,
                         threshold_coeffcient: float = 10.0,
                         frame_duration: float = 0.02) -> list[str]:
        """_summary_

        Args:
            audio_path (str): _description_
            transcription (list): _description_
            threshold_coeffcient (float, optional): _description_. 
            Defaults to 10.0.
        """
        # Load the audio file
        audio = auditok.load(audio_path)
        # Get the root mean square energy of the audio file
        biaudio, sr = librosa.load(audio_path, sr=None)
        rms = [-math.log(frame) for frame in 
               librosa.feature.rms(y=biaudio)[0] if frame != 0]
        rms_avg = sum(rms) / len(rms)
        
        # Split the audio file by energy level
        segments = auditok.split(
            audio, min_dur=0.1, max_dur=10, max_silence=0, 
            energy_threshold=rms_avg * threshold_coeffcient
        )

        word_intervals = []
        for _, segment in enumerate(list(segments)):
            start_time = round(segment.meta.start * 100) / 100
            end_time = round(segment.meta.end * 100) / 100
            word_intervals.append((start_time, end_time))
        
        parsed_transcription = [] # list of phonemes or words(with space)
        frame_middle_time = [frame_duration/2 + frame_duration * i for i in 
                             range(len(transcription))]
        idx_interval, idx_frame = 0, 0

        while idx_frame < len(frame_middle_time):
            # finish all words
            if idx_interval == len(word_intervals):
                while idx_frame < len(frame_middle_time):
                    parsed_transcription.append(str(transcription[idx_frame]))
                    idx_frame += 1
                break
            # the frame is before the start of the word
            if frame_middle_time[idx_frame] < word_intervals[idx_interval][0]:
                parsed_transcription.append(str(transcription[idx_frame]))
                idx_frame += 1
            
            # the frame is in the word
            elif frame_middle_time[idx_frame] >= \
                word_intervals[idx_interval][0] and \
                    frame_middle_time[idx_frame] <= \
                        word_intervals[idx_interval][1]:
                
                phones_in_word = []
                
                # collect all phonemes in the word
                while idx_frame < len(frame_middle_time) and \
                    frame_middle_time[idx_frame] <= \
                        word_intervals[idx_interval][1]:
                    phones_in_word.append(transcription[idx_frame])
                    idx_frame += 1
                word = " ".join([str(phone) for phone in phones_in_word])
                parsed_transcription.append("w" + word)
                idx_interval += 1
            
            # phone center directly pass the word duration
            else:
                idx_interval += 1
        
        merged_parsed_transcription, frame_counts = merge_phones_and_count_frames(parsed_transcription)
        
        start_ts = 0.0
        segments = []
        duration_per_frame = duration / sum(frame_counts)
        for count in frame_counts:
            segment_duration = count * duration_per_frame
            end_ts = start_ts + segment_duration
            segments.append([round(start_ts, 6), round(end_ts, 6)])
            start_ts = end_ts
        
        return merged_parsed_transcription, word_intervals, segments
    
    def _merge_transription_get_ts(self):
        print("Start merging transcriptions and getting timestamps...")
        start_time = time.time()
        for audio_name, transcription in tqdm(self.transcriptions.items()):
            duration = self.metadata[audio_name]["duration"]
            total_phones = len(transcription)
            num_frames_per_phone = list()
            merged_transcription = list()
            phone_ts = list()
            
            if total_phones == 0:
                self.metadata[audio_name]["transcription"] = []
                self.metadata[audio_name]["phone_ts"] = []
                continue
            
            num_frames = 1
            merged_transcription.append(transcription[0])
            for i in range(1, total_phones):
                if transcription[i] == transcription[i-1]:
                    num_frames += 1
                else:
                    num_frames_per_phone.append(num_frames)
                    num_frames = 1
                    merged_transcription.append(transcription[i])
            num_frames_per_phone.append(num_frames)
            
            # make sure we did not lose any frames
            assert sum(num_frames_per_phone) == total_phones, \
                f"Total phones: {total_phones}, sum of frames: \
                    {sum(num_frames_per_phone)}"
            assert len(num_frames_per_phone) == len(merged_transcription), \
                f"Length of num_frames_per_phone: {len(num_frames_per_phone)},\
                Length of merged_transcription: {len(merged_transcription)}"
                
            # get the timestamp of each phone
            accumulated_frames = 0
            frame_duration = duration / total_phones
            for i in range(len(merged_transcription)):
                phone_ts.append([accumulated_frames * frame_duration,])
                accumulated_frames += num_frames_per_phone[i]
                phone_ts[-1].append(accumulated_frames * frame_duration)
            
            self.metadata[audio_name]["transcription"] = merged_transcription
            self.metadata[audio_name]["phone_ts"] = phone_ts
        
        print(f"Transcriptions merged and timestamps got. \
            And used time: {time.time() - start_time:.2f} seconds.")
    
    def find_words(self, threshold_coeffcient: float):
        print("Finding words by energy...")
        start_time = time.time()
        for audio_name, transcription in tqdm(self.transcriptions.items()):
            audio_path = self.metadata[audio_name]["audio_path"]
            duration = self.metadata[audio_name]["duration"]
            try:
                parsed_transcription, word_intervals, phone_ts = self._parse_by_energy(
                    audio_path, duration, transcription, threshold_coeffcient)
                self.metadata[audio_name]["parsed_transcription"] = parsed_transcription
                self.metadata[audio_name]["word_intervals"] = word_intervals
                self.metadata[audio_name]["phone_ts"] = phone_ts
            except Exception as e:
                print(f"Error: {e}")
                print(f"Audio path: {audio_path}")
                print(f"filename: {audio_name}")
                print(f"Transcription: {transcription}")
        print(f"Words found. And used time: {time.time() - start_time:.2f} seconds.")
        json.dump(self.metadata, open(os.path.join(self.result_dir, "metadata_trans.json"), "w"))


def get_energy_word_cnt_dict():
    # Load all audio files and corresponding transcriptions
    parent_dir = "/bigdrive/dog/wav_audios"
    audio_path_file_path = "/home/sinong/ALP/DogRecognitionLabeling/grammar_induction/PyAdaGram/short-dog/train_audio.dat"
    raw_transcription_file_path = "short-dog/train_ori.dat"
    with open(audio_path_file_path, "r") as f:
        audio_paths = f.readlines()
    with open(raw_transcription_file_path, "r") as f:
        raw_transcriptions = f.readlines()
    audio_paths = [os.path.join(parent_dir, audio_path.strip()) for audio_path in audio_paths]
    transcriptions = [raw_transcription.strip().split() for raw_transcription in raw_transcriptions]
    
    colmns = ["word", "ori_audio_filename", "ori_audio_file_path", "start_time", "end_time"]
    df = pd.DataFrame(columns=colmns)
    
    # Parse the audio files by energy level
    all_words_cnt = defaultdict(int)
    for audio_path, transcription in tqdm(zip(audio_paths, transcriptions)):
        word_list, words_timestamps = parse_by_energy(audio_path, transcription)
        for word in word_list:
            all_words_cnt[word] += 1
        new_data = pd.DataFrame(words_timestamps, columns=colmns)
        df = pd.concat([df, new_data], ignore_index=True)
        print(df)
        
    # Save the timestamp DataFrame as a CSV file
    df.to_csv("energy_word_timestamps.csv", index=False)
    
    # Save the word count dictionary as a JSON file
    with open("energy_word_cnt_dict.json", "w") as f:
        json.dump(all_words_cnt, f)
    
    return all_words_cnt    
    
    
def get_energy_transcription_list():
    '''
    The transcription list is parsed by energy level.
    For example: Each word is represented by a string starting with 'w' and followed by the phonemes.
    The string
    ['w39 137 103 101', '137', 'w113 59 43 6 8 92', '104', '39', '4', '100', '89', '91', '39', '4', '39', '4']
    '''
    # Load all audio files and corresponding transcriptions
    parent_dir = "/bigdrive/dog/wav_audios"
    audio_path_file_path = "/home/sinong/ALP/DogRecognitionLabeling/grammar_induction/PyAdaGram/short-dog/train_audio.dat"
    raw_transcription_file_path = "short-dog/train_ori.dat"
    with open(audio_path_file_path, "r") as f:
        audio_paths = f.readlines()
    with open(raw_transcription_file_path, "r") as f:
        raw_transcriptions = f.readlines()
    audio_paths = [os.path.join(parent_dir, audio_path.strip()) for audio_path in audio_paths]
    transcriptions = [raw_transcription.strip().split() for raw_transcription in raw_transcriptions]
    
    # Parse the audio files by energy level
    all_transcriptions = []
    all_timestamps = []
    for audio_path, transcription in tqdm(zip(audio_paths, transcriptions)):
        parsed_transcription, word_intervals = parse_by_energy2(audio_path, transcription)
        all_transcriptions.append(parsed_transcription)
        all_timestamps.append(word_intervals)
    
    # Save the word count dictionary as a JSON file
    with open("energy_word_transcription.json", "w") as f:
        json.dump(all_transcriptions, f)
        
    with open("energy_word_timestamps.json", "w") as f:
        json.dump(all_timestamps, f)
    
    return all_transcriptions  


def find_words_by_energy(word_intervals: list[tuple],
                         transcription: list[str],
                         frame_duration: float = 0.02,):
    '''
    This is a redundant function.
    It is used for testing the parse_by_energy function.
    '''
    parsed_transcription = [] # list of phonemes or words(with space)
    frame_middle_time = [frame_duration/2 + frame_duration * i for i in 
                            range(len(transcription))]
    idx_interval, idx_frame = 0, 0
    while idx_frame < len(frame_middle_time):
        # finish all words
        if idx_interval == len(word_intervals):
            break
        # the frame is before the start of the word
        if frame_middle_time[idx_frame] < word_intervals[idx_interval][0]:
            parsed_transcription.append(transcription[idx_frame])
            idx_frame += 1
        
        # the frame is in the word
        elif frame_middle_time[idx_frame] >= \
            word_intervals[idx_interval][0] and \
                frame_middle_time[idx_frame] <= \
                    word_intervals[idx_interval][1]:
            phones_in_word = []
            
            # collect all phonemes in the word
            while idx_frame < len(frame_middle_time) and \
                frame_middle_time[idx_frame] < \
                    word_intervals[idx_interval][1]:
                phones_in_word.append(transcription[idx_frame])
                idx_frame += 1
            word = " ".join(merge_consecutive_transcriptions(
                phones_in_word))
            parsed_transcription.append("w" + word)
            idx_interval += 1
        
        # phone center directly pass the word duration
        else:
            idx_interval += 1
    print(parsed_transcription)
    return parsed_transcription, word_intervals


def merge_phones_and_count_frames(phone_list):
    merged_phones = []
    frame_counts = []
    last_phone = None
    current_count = 0
    
    def process_word(word):
        ''' Helper function to merge consecutive phones in a word (starting with w). '''
        phones = word[1:].split()  # Remove "w" and split the word into phones
        merged_word = []
        frame_count_word = []
        last_phone_in_word = None
        current_word_count = 0
        
        for phone in phones:
            if phone == last_phone_in_word:
                current_word_count += 1
            else:
                if last_phone_in_word is not None:
                    merged_word.append(last_phone_in_word)
                    frame_count_word.append(current_word_count)
                last_phone_in_word = phone
                current_word_count = 1
        
        # Append the last phone in the word
        merged_word.append(last_phone_in_word)
        frame_count_word.append(current_word_count)
        
        # Join the merged phones back into a word with "w"
        merged_word_str = "w" + " ".join(merged_word)
        return merged_word_str, frame_count_word
    
    for element in phone_list:
        if element.startswith("w"):
            # Process word starting with w
            merged_word, word_frame_counts = process_word(element)
            merged_phones.append(merged_word)
            frame_counts.extend(word_frame_counts)
            last_phone = None  # Reset last_phone after processing a word
        else:
            # Process normal integer strings
            if element == last_phone:
                current_count += 1
            else:
                if last_phone is not None:
                    merged_phones.append(last_phone)
                    frame_counts.append(current_count)
                last_phone = element
                current_count = 1

    # Append the last phone after the loop
    if last_phone is not None:
        merged_phones.append(last_phone)
        frame_counts.append(current_count)
    
    return merged_phones, frame_counts


if __name__ == "__main__":
    # get_energy_word_cnt_dict()
    # get_energy_transcription_list()
    # finder = WordFinder("/data/xingyuan/animal_language/v2/out/fairseq/"
    #                     "six_balance/work/iter2_130k_11th_km250")
    # finder.find_words(10.0)
    # test the parse_by_energy function
    # segments = [(0.00, 0.06), (0.07, 0.071), (0.13, 0.16), (0.17, 0.20)]
    # transcription = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"]
    # find_words_by_energy(segments, transcription)
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_dir", type=str, required=True)
    parser.add_argument("--threshold_coeffcient", type=float, default=10.0)
    args = parser.parse_args()
    finder = WordFinder(args.result_dir)
    finder.find_words(args.threshold_coeffcient)
    