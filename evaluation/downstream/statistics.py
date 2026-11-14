import os
import numpy as np
import pandas as pd
import librosa

from collections import Counter


feature_type_mapping = {
    "mfcc": "00MFCC",
    "hubert-250-no-mapping": "01Hubert-250-no-mapping",
    "hubert-250-cc-no-mapping": "02Hubert-250-cc-no-mapping",
    "hubert-250-cc-hubert-250-t5-p5": "03Hubert-250-cc-hubert-250-t5-p5",
    "hubert-250-cc-hubert-250-t5-p5-only": "04Hubert-250-cc-hubert-250-t5-p5-only",
    "hubert-250-cc-hubert-250-t5-p10-only": "05Hubert-250-cc-hubert-250-t5-p10-only",
    "hubert-250-cc-hubert-250-t5-p15-only": "06Hubert-250-cc-hubert-250-t5-p15-only",
    "hubert-500-no-mapping": "07Hubert-500-no-mapping",
    "hubert-500-cc-no-mapping": "08Hubert-500-cc-no-mapping",
    "hubert-500-cc-hubert-500-t5-p5": "09Hubert-500-cc-hubert-500-t5-p5",
    "hubert-500-cc-hubert-500-t5-p5-only": "10Hubert-500-cc-hubert-500-t5-p5-only",
    "hubert-500-cc-hubert-500-t5-p10-only": "11Hubert-500-cc-hubert-500-t5-p10-only",
    "hubert-500-cc-hubert-500-t5-p15-only": "12Hubert-500-cc-hubert-500-t5-p15-only",
}


def get_stat_from_folder_audios(folder_path: str):
    """
    Get the statistics of the audio files in the folder
    """
    audio_filenames = list(os.listdir(folder_path))
    audio_duration = []
    for audio_file in audio_filenames:
        audio_path = os.path.join(folder_path, audio_file)
        audio, sr = librosa.load(audio_path)
        audio_duration.append(len(audio) / sr)
    
    audio_durations = np.array(audio_duration)
    
    avg_duration = np.mean(audio_durations)
    std_duration = np.std(audio_durations)
    max_duration = np.max(audio_durations)
    min_duration = np.min(audio_durations)
    median_duration = np.median(audio_durations)
    num_audios = len(audio_filenames)
    q25, q75 = np.percentile(audio_durations, [25, 75])
    
    print(f"Number of audios: {num_audios}")
    print(f"Average duration: {avg_duration}")
    print(f"Standard deviation of duration: {std_duration}")
    print(f"Max duration: {max_duration}")
    print(f"Min duration: {min_duration}")
    print(f"Median duration: {median_duration}")
    print(f"25% quantile duration: {q25}")
    print(f"75% quantile duration: {q75}")


def analyze_dt_results(folder_path: str, file_num: int = 13):
    csv_filelist = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.csv')]
    all_data_list = []
    for csv_file in csv_filelist:
        if "analysis" in csv_file:
            continue
        filename = os.path.basename(csv_file)[:-4]
        feature_type = "-".join(filename.split('_')[:-2])
        feature_type = feature_type_mapping[feature_type]

        data = pd.read_csv(csv_file)
        data.insert(0, 'feature_type', feature_type)
        all_data_list.append(data)

    # Concatenate all data
    all_data = pd.concat(all_data_list)
    # only leave the rows with label "macro avg" and "weighted avg"
    filtered_df = all_data[all_data['Label'].isin(['macro avg', 'weighted avg'])]
    sorted_df = filtered_df.sort_values(by=['Label', 'Model', 'feature_type'])
    
    new_data = []
    champion_list = []
    for i in range(0, len(sorted_df), file_num):
        new_data.append(sorted_df.iloc[i:i+file_num])
        
        block = sorted_df.iloc[i:i+file_num]
        second_largest_row = block.nlargest(4, 'F1-Score')
        champion_list.append(second_largest_row)
        
        empty_row = pd.DataFrame([[np.nan] * len(sorted_df.columns)], columns=sorted_df.columns)
        new_data.append(empty_row)
    
    champion_df = pd.concat(champion_list)
    print(Counter(champion_df['feature_type'].tolist()))
    
    new_df = pd.concat(new_data, ignore_index=True)
    
    # Save the sorted dataframe to a csv file
    new_df.to_csv(os.path.join(folder_path, 'analysis.csv'), index=False)
  
    
if __name__ == "__main__":
    # folder_name = "/data/dogs/sixbarktype_dataset/audios/unbalanced"
    # get_stat_from_folder_audios(folder_name)
    folder_path = "/home/sinong/ALP/AniLang/Evaluation/checkpoints/barktype/single"
    # folder_path = "/home/sinong/ALP/AniLang/Evaluation/checkpoints/activity"
    analyze_dt_results(folder_path, file_num=7)
    