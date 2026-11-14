__version__ = "0.1.0"
__author__ = "Sinong Wang"

import os
import random
import datetime
import json
import shutil

import librosa
import matplotlib.pyplot as plt

from collections import defaultdict
from itertools import combinations
from functools import partial

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from utils import calculate_euclidean_distance
from utils import calculate_cosine_similarity
from utils import calculate_dtw_distance
from utils import calculate_list_accuracy

    
class PhoneEvaluator:
  '''
  This class is used to evaluate the quality of phones.
  It doesn't have the functionality to generate phone samples.
  The input to this class should be a folder with phone samples.
  - phones_samples_folder: the folder with phone samples
    - phone1_folder
      - phone1_sample1.wav
      - phone1_sample2.wav
      - ...
    - phone2_folder
      - phone2_sample1.wav
      - phone2_sample2.wav
      - ...
    - ...
    
  This class can:
  1. Generate phone quality evaluation pairs 
  for human evaluation or automatic evaluation.
  2. Auto evaluate the quality of phones 
  based on the phone samples it made.
  3. Plot the evaluation results include 
  confusion matrix and histrogram.
  
  Data Structure & Examples:
  self.phone_samples_folder: the folder with phone samples 
    - "/path/to/phone_samples"
  self.phone_list: the list of phones in the phone samples folder 
    - ['phone1', 'phone2', ...]
  self.phone_sample_path_dict: the dictionary with phone as key 
  and the list of phone sample paths as value
    - {'phone1': ['/path/to/phone_samples/phone1/phone1_sample1.wav', 
    '/path/to/phone_samples/phone1/phone1_sample2.wav', ...], ...}
  self.phone_sample_number_dict: the dictionary with phone as key 
  and the number of phone samples as value
    - {'phone1': 10, 'phone2': 10, ...}
  
  '''
  def __init__(self, phone_samples_folder: str):
    self.phone_samples_folder = phone_samples_folder
    self.phone_list = os.listdir(phone_samples_folder)
    self.phone_sample_path_dict = defaultdict(list)
    self.phone_sample_number_dict = defaultdict(int)
    self._scan_phone_samples()
    
    self.DISTANCE_METRICS = ['euclidean', 'cosine', 'dtw']
      
  def _scan_phone_samples(self):
    '''
    Scan the phone samples folder and store the phone samples 
    in the phone_sample_path_dict.
    Scan the phone samples folder and store the number of phone samples 
    in the phone_sample_number_dict.
    '''
    for phone in self.phone_list:
        phone_folder = os.path.join(self.phone_samples_folder, phone)
        for sample_filename in os.listdir(phone_folder):
            sample_path = os.path.join(phone_folder, sample_filename)
            self.phone_sample_path_dict[phone].append(sample_path)
            self.phone_sample_number_dict[phone] += 1
  
  def generate_evaluation_samples(self, 
                                  evaluatioin_mode: str, 
                                  total_sample_number: int, 
                                  save_path: str):
    '''
    Generate phone quality evaluation pairs for human evaluation 
    or automatic evaluation.
    - evaluatioin_mode: 'abx' or 'ab?'
        - 'abx': generate same/different pairs for human evaluation
        - 'ab?': generate a different pair and given a phone,
        find which one is the same as the given phone
    - total_sample_number: the total number of evaluation pairs
    - save_path: the path to save the evaluation pairs
    '''
    if evaluatioin_mode == 'abx':
      # make sure there are at least 2 samples for each phone
      assert all(self.phone_sample_number_dict[phone] >= 2 \
        for phone in self.phone_list),\
          "The number of samples for each phone should be at least 2."
      
      # generate different pairs
      different_pairs = list(combinations(self.phone_list, 2))
      # deal with the case when the total_sample_number 
      # is larger than the number of different pairs
      if total_sample_number > len(different_pairs):
        enlarge_coefficient = total_sample_number // len(different_pairs) + 1
        different_pairs = different_pairs * enlarge_coefficient
      pairs_to_generate = random.sample(different_pairs, total_sample_number)
      
      # sample instances for each pair
      evaluation_samples_metadata = []
      evaluation_answer = []
      for pair in pairs_to_generate:
        # decide which phone used twice
        twice_phone_idx = 0 if random.random() < 0.5 else 1
        twice_phone = pair[twice_phone_idx]
        other_phone = pair[1 - twice_phone_idx]
        
        # sample two instances for the twice phone
        twice_phone_instances = random.sample(
          self.phone_sample_path_dict[twice_phone], 2)
        
        # sample one instance for the other phone
        other_phone_instance = random.choice(
          self.phone_sample_path_dict[other_phone])
        
        # add the evaluation pair
        answer_idx = 0 if random.random() < 0.5 else 1
        wrong_idx = 1 - answer_idx
        evaluation_samples_metadata.append(
          {'answer': f'reference_phone_{answer_idx}', 
          f'reference_phone_{answer_idx}_number': twice_phone,
          f'reference_phone_{wrong_idx}_number': other_phone,
          f'reference_phone_{answer_idx}': twice_phone_instances[0], 
          f'reference_phone_{wrong_idx}': other_phone_instance, 
          'other_phone': twice_phone_instances[1]})
        evaluation_answer.append(answer_idx)
      
      # generate output folder
      date_time_string = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
      output_folder = os.path.join(save_path, 
                                   "triplet_pairs_phone_evaluation" 
                                   + date_time_string)
      os.makedirs(output_folder, exist_ok=True)
      
      # add metadata into the evaluation sample output folder
      metadata = {'sample_path': self.phone_samples_folder,
                  'answer': evaluation_answer,
                  'metadata': list(enumerate(evaluation_samples_metadata))}
      metadata_path = os.path.join(output_folder, "metadata.json")
      with open(metadata_path, 'w') as f:
        json.dump(metadata, f)
      print(f"Metadata is saved to {metadata_path}")
      
      # save the evaluation samples
      print(f"Start saving the evaluation samples to {output_folder}")
      for idx, meta in enumerate(evaluation_samples_metadata):
        sample_output_folder = os.path.join(output_folder, f"sample_{idx}")
        os.makedirs(sample_output_folder, exist_ok=True)
        shutil.copy(meta['reference_phone_0'], 
                    os.path.join(sample_output_folder,
                                 "reference_phone_0.wav"))
        shutil.copy(meta['reference_phone_1'], 
                    os.path.join(sample_output_folder,
                                 "reference_phone_1.wav"))
        shutil.copy(meta['other_phone'], 
                    os.path.join(sample_output_folder, "target_phone.wav"))
      print(f"Samples are saved to {output_folder}")
      
  def evaluate_human_result(self, evaluation_samples_folder: str,
                            prediction_file_path: str):
    '''
    Auto evaluate the quality of phones 
    based on the phone samples it made.
    - evaluation_samples_folder: the folder with evaluation samples
    - result_folder: the folder to save the evaluation results
    '''
    metadata_path = os.path.join(evaluation_samples_folder, "metadata.json")
    with open(metadata_path, 'r') as f:
      metadata = json.load(f)
    ground_truths = metadata['answer']
    
    predictions = []
    with open(prediction_file_path, 'r') as f:
      # read txt file
      lines = f.readlines()
      for line in lines:
        for char in line.strip():
          predictions.append(int(char))
        
    assert len(predictions) == len(ground_truths), \
      "The length of predictions and ground truths should be the same."
    
    accuracy = calculate_list_accuracy(predictions, ground_truths)
    pred_filename = os.path.basename(prediction_file_path)
    print(f"The accuracy of the human evaluation {pred_filename}"
          f"is {accuracy}.")
  
  def evaluate_by_acoustic_fea(self, 
                       evaluation_samples_folder: str,
                       sample_mode: str,
                       distance_metric: str,
                       feature_type: str,
                       save_result: bool = False,
                       plot_confusion_matrix: bool = False,
                       result_folder: str = None):
    '''
    Evaluate the quality of phones based on the MFCC features 
    of the phone samples.
    '''
    predictions = []
    ground_truths = list()
    
    # decide the distance function
    if distance_metric == 'euclidean':
      distance_function = calculate_euclidean_distance
    elif distance_metric == 'cosine':
      distance_function = calculate_cosine_similarity
    elif distance_metric == 'dtw':
      distance_function = calculate_dtw_distance
    else:
      raise ValueError("The distance metric should be 'euclidean', \
                       'cosine' or 'dtw'.")
    distance_function = partial(distance_function, feature_type=feature_type)
    
    # load the ground truth
    # the metadata file should be in the evaluation samples folder
    metadata_path = os.path.join(evaluation_samples_folder, "metadata.json")
    with open(metadata_path, 'r') as f:
      metadata = json.load(f)
    ground_truths = metadata['answer']
    
    # load the evaluation samples
    sample_folders = [
    item for item in os.listdir(evaluation_samples_folder)
    if os.path.isdir(
        os.path.join(evaluation_samples_folder, item)
    )]
    for sample_folder in sample_folders:
      sample_folder_path = os.path.join(evaluation_samples_folder, 
                                        sample_folder)
      
      if sample_mode == 'abx':
        reference_phone_0_path = os.path.join(sample_folder_path, 
                                              "reference_phone_0.wav")
        reference_phone_1_path = os.path.join(sample_folder_path, 
                                              "reference_phone_1.wav")
        target_phone_path = os.path.join(sample_folder_path, 
                                         "target_phone.wav")
        
        distance_ref_0_target = distance_function(reference_phone_0_path, 
                                                  target_phone_path)
        distance_ref_1_target = distance_function(reference_phone_1_path,
                                                  target_phone_path)
        # predict the phone with the smallest distance
        prediction = 0 if distance_ref_0_target < distance_ref_1_target else 1
        predictions.append(prediction)
        
      elif sample_mode == 'ab?':
        pass
      
      else:
        raise ValueError("The sample mode should be 'abx' or 'ab?'.")
    
    # save the evaluation results
    if save_result:
      result_filename = f"evaluation_result_MFCC_{distance_metric}.json"
      result_path = os.path.join(result_folder, result_filename)
      with open(result_path, 'w') as f:
        json.dump(predictions, f)
      print(f"The evaluation result is saved to {result_path}.")
    
    # calculate the prediction accuracy
    accuracy = calculate_list_accuracy(predictions, ground_truths)
    print(f"The accuracy of the evaluation by MFCC with {distance_metric}",
          f"distance is {accuracy}.")
    
    # plot the confusion matrix
    if plot_confusion_matrix:
      title = f"Confusion Matrix of MFCC with {distance_metric} Distance"
      save_file_path = os.path.join(result_folder, 
                                    f"confusion_matrix_MFCC_\
                                      {distance_metric}.png")
      self._plot_confusion_matrix(predictions, ground_truths, 
                                  title, save_file_path)
      print(f"The confusion matrix is saved to {save_file_path}.")
    
  def evaluate_by_acoustic_fea_all_metrics(self,
                                   evaluation_samples_folder: str,
                                   sample_mode: str,
                                   feature_type: str = 'mel',
                                   save_result: bool = False,
                                   plot_confusion_matrix: bool = False,
                                   result_folder: str = None):
    for distance_metric in self.DISTANCE_METRICS:
      self.evaluate_by_acoustic_fea(evaluation_samples_folder, 
                            sample_mode, 
                            distance_metric, 
                            feature_type,
                            save_result, 
                            plot_confusion_matrix, 
                            result_folder)
    return
    
  def _plot_confusion_matrix(self, predictions: list, 
                             ground_truths: list,
                             title: str,
                             save_file_path: str) -> None:
    if len(predictions) == 0 or len(ground_truths) == 0:
      print("There is no prediction or ground truth.")
      return
    
    assert len(predictions) == len(ground_truths), \
      "The length of predictions and ground truths should be the same."
    assert type(predictions[0]) == type(ground_truths[0]), \
      "The type of predictions and ground truths should be the same."

    cm = confusion_matrix(ground_truths, predictions)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title(title)
    plt.savefig(save_file_path)
      

def make_answer_file(start_number: int, end_number: int, filename: str):
  filename = os.path.join("/home/sinong/ALP/AniLang/Evaluation/checkpoints/phoneme_samples", filename)
  with open(filename, 'w') as f:
        for num in range(start_number, end_number + 1):
            f.write(f"{num}:\n")


if __name__ == "__main__":
  # phone_samples_folder = ("/home/sinong/ALP/AniLang/PhonemeFinding/checkpoints/phoneme_refined/hubert250-t5-p5-samples")
  # phone_evaluator = PhoneEvaluator(phone_samples_folder)
  
  # phone_evaluator.generate_evaluation_samples('abx', 210, 
  #     ("/home/sinong/ALP/AniLang/Evaluation/checkpoints/phoneme_samples/hubert-250-t5-p5"))
  
  make_answer_file(0, 104, "answer0-104.txt")
  make_answer_file(105, 209, "answer105-209.txt")
  make_answer_file(0, 99, "answer0-99.txt")
  make_answer_file(100, 199, "answer100-199.txt")
  make_answer_file(200, 299, "answer200-299.txt")
  make_answer_file(300, 399, "answer300-399.txt")
  make_answer_file(400, 499, "answer400-499.txt")
  
  # phone_evaluator.evaluate_by_acoustic_fea_all_metrics(
  #   evaluation_samples_folder=("/home/sinong/ALP/DogRecognitionLabeling/"
  #               "grammar_induction/PyAdaGram/phone_evaluation_tri_pairs/"
  #               "triplet_pairs_phone_evaluation20240910174727"),
  #   sample_mode='abx',
  #   feature_type='stft',)
  
  # phone_evaluator.evaluate_human_result(
  #   evaluation_samples_folder=("/home/sinong/ALP/DogRecognitionLabeling/"
  #               "grammar_induction/PyAdaGram/phone_evaluation_tri_pairs/"
  #               "triplet_pairs_phone_evaluation20240910174727"),
  #   prediction_file_path="essam_result.txt"
  # )
