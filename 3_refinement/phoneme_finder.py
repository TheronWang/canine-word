import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json, joblib, os, random, shutil, auditok
import multiprocessing as mp
from itertools import chain
from datetime import datetime
from collections import defaultdict
from itertools import combinations

from constant import cut_audio

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import librosa
from sklearn.metrics import mutual_info_score
from scipy.stats import entropy
from sklearn.manifold import TSNE



def kl_divergence(P, Q, epsilon=1e-10) -> float:
    """
    Calculate the Kullback-Leibler divergence 
    between two discrete distributions.
    
    Parameters:
    P (array-like): True probability distribution.
    Q (array-like): Approximating probability distribution.
    epsilon (float): Small value to avoid division by zero 
        or log of zero.
    
    Returns:
    float: KL divergence D_{KL}(P || Q).
    """
    P = np.array(P, dtype=np.float64) + epsilon
    Q = np.array(Q, dtype=np.float64) + epsilon
    
    P = P / np.sum(P)
    Q = Q / np.sum(Q)
    
    # Calculate the KL divergence
    kl_div = np.sum(P * np.log(P / Q))

    return kl_div


def calculate_kl_divergence(P: dict, Q: dict) -> float:
    '''
    Calculate the Kullback-Leibler divergence
    between two discrete distributions.
    '''
    P_prob, Q_prob = P.copy(), Q.copy()
    
    # combined set of events
    all_keys = set(P_prob.keys()).union(set(Q_prob.keys()))
    
    # create probability distributions
    P_prob = [P_prob.get(k, 0) for k in all_keys]
    Q_prob = [Q_prob.get(k, 0) for k in all_keys]
    
    kl_div = kl_divergence(P_prob, Q_prob)
    
    return kl_div


def calculate_js_divergence(P: dict, Q: dict) -> float:
    """
    Calculate the Jensen-Shannon divergence between two discrete distributions.
    
    Parameters:
    P (dict): True probability distribution.
    Q (dict): Approximating probability distribution.
    
    Returns:
    float: JS divergence D_{JS}(P || Q).
    """
    # convert counts to probabilities
    P_prob = {k: v / sum(P.values()) for k, v in P.items()}
    Q_prob = {k: v / sum(Q.values()) for k, v in Q.items()}
    
    # combined set of events
    all_keys = set(P_prob.keys()).union(set(Q_prob.keys()))
    
    # create probability distributions
    P_prob = [P_prob.get(k, 0) for k in all_keys]
    Q_prob = [Q_prob.get(k, 0) for k in all_keys]
    
    # calculate the Jensen-Shannon divergence
    M = 0.5 * (np.array(P_prob) + np.array(Q_prob))
    js_div = 0.5 * kl_divergence(P_prob, M) + 0.5 * kl_divergence(Q_prob, M)
    
    return js_div


class UnionFind:
    def __init__(self, elements):
        self.parent = {e: e for e in elements}
        self.rank = {e: 0 for e in elements}

    def find(self, item):
        if self.parent[item] != item:
            self.parent[item] = self.find(self.parent[item])
        return self.parent[item]

    def union(self, set1, set2):
        root1 = self.find(set1)
        root2 = self.find(set2)

        if root1 != root2:
            if self.rank[root1] > self.rank[root2]:
                self.parent[root2] = root1
            elif self.rank[root1] < self.rank[root2]:
                self.parent[root1] = root2
            else:
                self.parent[root2] = root1
                self.rank[root1] += 1

def merge_sets(sets):
    # Collect all unique elements
    elements = set()
    for s in sets:
        elements.update(s)
    
    # Initialize UnionFind with all unique elements
    uf = UnionFind(elements)
    
    # Union sets that have common elements
    for s in sets:
        s = list(s)
        for i in range(len(s) - 1):
            uf.union(s[i], s[i + 1])
    
    # Collect merged sets
    merged = {}
    for e in elements:
        root = uf.find(e)
        if root not in merged:
            merged[root] = set()
        merged[root].add(e)
    
    return list(merged.values())


# Helper function moved outside the class to avoid pickling issues
def process_transcriptions(transcriptions, parse_mode):
    local_word_count = defaultdict(int)
    if parse_mode == "energy":
        for transcription in transcriptions:
            for word in transcription:
                if word[0] == "w":
                    local_word_count[word[1:]] += 1
    elif parse_mode == "grammar":
        for transcription in transcriptions:
            for word in transcription:
                local_word_count[word] += 1
    return local_word_count


# Helper function moved outside the class to avoid pickling issues
def process_transcription_chunk(transcriptions_chunk, replacement_mapping):
    for idx_sentence, transcription in enumerate(transcriptions_chunk):
        for idx_word, word in enumerate(transcription):
            if word.startswith("w"):
                phones = word[1:].split()
                for i in range(len(phones)):
                    for phone, replacement in replacement_mapping.items():
                        if phones[i] == phone:
                            phones[i] = replacement
                new_word = "w" + " ".join(phones)
                transcription[idx_word] = new_word
            else:
                if word in replacement_mapping.keys():
                    transcription[idx_word] = replacement_mapping[word]
        transcriptions_chunk[idx_sentence] = transcription
    return transcriptions_chunk


# Helper function moved outside the class to avoid pickling issues
# TODO: Merge context phones
def process_transcription_deduplication(transcriptions_chunk):
    for idx_sentence, transcription in enumerate(transcriptions_chunk):
        for idx_word, word in enumerate(transcription):
            if not word.startswith("w"):
                continue
            new_word = []
            for phone in word[1:].split():
                if not new_word or phone != new_word[-1]:
                    new_word.append(phone)
            new_word = "w" + " ".join(new_word)
            transcription[idx_word] = new_word
        transcriptions_chunk[idx_sentence] = transcription
    return transcriptions_chunk


class PhonemeFinder():
    def __init__(self, parse_mode: str, parsed_transcription_file_path: str, valid_word_cnt_threshold: int=1, eliminate_noise_manually: bool=False, percentile: int=5):
        self.PARSE_MODES = ["energy", "grammar"]
        self.CONTEXT_MODES = ["phoneme", "word", "ngram"]
        self.NOISE_PHONES = [4, 10, 20, 23, 24, 32, 37, 39, 50, 52, 53, 54, 65, 71, 72, 78, 79, 82, 85, 87, 89, 108, 109, 114, 119, 120, 122, 134, 137]
        self.CK_PATH = "/home/sinong/ALP/AniLang/PhonemeFinding/checkpoints/phoneme_refined/" + datetime.now().strftime("%y%m%d_%H%M%S")
        
        self.percentile = percentile
        self.valid_word_cnt_threshold = valid_word_cnt_threshold
        self.eliminate_noise_manually = eliminate_noise_manually
        self.parse_mode = parse_mode
        self.parsed_transcription_file_path = parsed_transcription_file_path
        self.all_words_cnt = defaultdict(int)
        self.all_word_list = list()
        self.valid_word_list = list() # words that have more than word_cnt_threshold occurrences
        self.valid_word_list_without_noise = list() # words that have more than word_cnt_threshold occurrences and do not contain noise phones
        self.all_word_prior_prob = dict() # prior probability of each word
        
        self.found_phonemes = set()
        self.found_words = defaultdict(int)
        self.minimal_pairs = defaultdict(list)
        self.word_context = defaultdict(lambda: defaultdict(int))
        self.mapping_phoneme = dict()
        self.this_round_mapping_phoneme = dict()
        
        self.percentile_95_list = list()
        self.percentile_75_list = list()
        self.percentile_50_list = list()
        self.percentile_25_list = list()
        self.percentile_5_list = list()
        
        # build checkpoint folder
        os.makedirs(self.CK_PATH, exist_ok=True)
        
        self.read_transcription(self.parsed_transcription_file_path)
        self.get_word_list_and_word_cnt_dict()
        if self.eliminate_noise_manually:
            self.eliminate_words_containing_noise()
        # self.prepare_all_phone_plot()
    
    def read_transcription(self, transcription_file_path: str):
        '''
        The transcription file should be a json file with the following format:
        [["32 23 221", "3", ...], ["21 33 1", "125 22", ...], ...]
        or
        [["w32 23 221", "3", ...], ["w21 33 1", "w125 22", ...], ...]
        '''
        print("Start Loading Transcription File...")
        start_time = datetime.now()
        
        with open(transcription_file_path, 'r') as f:
            metadata = json.load(f)
        transcriptions = list()
        phone_ts = list()
        audio_paths = list()
        raw_transcriptions = list()
        
        self.samples = defaultdict(list)
        
        for _, data in metadata.items():
            try:
                transcriptions.append(data["parsed_transcription"])
                phone_ts.append(data["phone_ts"])
                audio_paths.append(data["audio_path"])
                raw_transcriptions.append([str(phone) for phone in data["transcription"]])
                
                parsed_transcription = data["parsed_transcription"]
                ts = data["phone_ts"]
                merged_phones = []
                for word in parsed_transcription:
                    if word.startswith("w"):
                        merged_phones.extend(word[1:].split())
                    else:
                        merged_phones.extend(word.split())
                
                for (start_time, end_time), phone in zip(ts, merged_phones):
                    if end_time - start_time > 0.10:
                        self.samples[phone].append({"audio_path": data["audio_path"], "start_time": start_time, "end_time": end_time})
                
                
            except:
                pass
        print("smallest number of samples: ", sum([len(v) > 4 for v in self.samples.values()]))
        
        # sample_save_path = "/home/sinong/ALP/AniLang/PhonemeFinding/checkpoints/phoneme_refined/phone_samples"
        # os.makedirs(sample_save_path, exist_ok=True)
        # for phone, samples in self.samples.items():
        #     phone_folder = os.path.join(sample_save_path, str(phone))
        #     os.makedirs(phone_folder, exist_ok=True)
        #     sample_number = min(len(samples), 10)
        #     selected_samples = random.sample(samples, sample_number)
        #     for sample in selected_samples:
        #         audio_path = sample["audio_path"]
        #         start_time = sample["start_time"]
        #         end_time = sample["end_time"]
        #         output_path = os.path.join(phone_folder, f"{os.path.basename(audio_path)}_{start_time}_{end_time}.wav")
        #         cut_audio(audio_path, start_time, end_time, output_path)
            
            
        assert len(transcriptions) == len(phone_ts) == len(audio_paths) \
            == len(raw_transcriptions), \
            f"The number of transcriptions, phone_ts, and audio_paths should \
                be the same. len(transcriptions): {len(transcriptions)}, \
                len(phone_ts): {len(phone_ts)}, len(audio_paths): \
                {len(audio_paths)}, len(raw_trans): {len(raw_transcriptions)}."
        
        self.transcriptions = transcriptions
        self.phone_ts = phone_ts
        self.audio_paths = audio_paths
        self.raw_transcriptions = raw_transcriptions

        print("Transcription File Loaded Successfully.")
        print("Time Cost: ", datetime.now() - start_time, " seconds.")
        print()
            
    def get_word_list_and_word_cnt_dict(self):
        '''
        This function is used to get the word list and word count dictionary.
        The word list is the list of words that have more than word_cnt_threshold occurrences.
        '''
        print("Start Analize Words, Word Count and prior probability...")
        start_time = datetime.now()
        # Split transcriptions into chunks
        num_cores = mp.cpu_count()
        chunk_size = len(self.transcriptions) // num_cores
        transcription_chunks = [self.transcriptions[i:i + chunk_size] for i in range(0, len(self.transcriptions), chunk_size)]
        
        # Create a pool of workers and process the chunks in parallel
        with mp.Pool(processes=num_cores) as pool:
            results = pool.starmap(process_transcriptions, [(chunk, self.parse_mode) for chunk in transcription_chunks])

        # Merge the results from all processes
        self.all_words_cnt = defaultdict(int)
        for result in results:
            for word, cnt in result.items():
                self.all_words_cnt[word] += cnt
        
        # Filter out words based on valid_word_cnt_threshold
        self.valid_word_list = [word for word, cnt in self.all_words_cnt.items() if cnt >= self.valid_word_cnt_threshold]
        self.found_words = {word: cnt for word, cnt in self.all_words_cnt.items() if word in self.valid_word_list}
        self.found_phonemes = {phone for word in self.found_words for phone in word.split()}
        
        # Get all word list and word prior probability
        self.all_word_list = list(self.all_words_cnt.keys())
        total_word_count = sum(self.all_words_cnt.values())
        self.all_word_prior_prob = {word: cnt / total_word_count for word, cnt in self.all_words_cnt.items()}
        
        print("Words, Word Count and Prior Probability Analized Successfully.")
        print("Time Cost: ", datetime.now() - start_time, " seconds.")
        print()

    def save_config(self):
        filename = os.path.join(self.CK_PATH, "config.json")
        config = {
            "valid_word_cnt_threshold": self.valid_word_cnt_threshold,
            "eliminate_noise_manually": self.eliminate_noise_manually,
            "parse_mode": self.parse_mode,
            "percentile": self.percentile,
            "found_phonemes": list(self.found_phonemes),
            "found_words": self.found_words,
            "mapping_phoneme": self.mapping_phoneme,
        }
        with open(filename, 'w') as f:
            json.dump(config, f)
  
    def eliminate_words_containing_noise(self):
        '''
        This function is used to eliminate the words that contain noise phones.
        '''
        self.valid_word_list = [word for word in self.valid_word_list if not any([int(phone) in self.NOISE_PHONES for phone in word.split()])]
    
    def prepare_all_phone_plot(self):
        cluster_feature_file_path = "60k_11th_km140.pkl"
        with open(cluster_feature_file_path, 'rb') as f:
            cluster_feature = joblib.load(f)
        c_np = cluster_feature.cluster_centers_
        
        self.plot_label = list(range(140))
        
        # Plot t-SNE of the cluster centers
        
        tsne = TSNE(n_components=2)
        self.c_tsne = tsne.fit_transform(c_np)
        
        # plot the t-SNE in 2D of the idx_print_phonemes
        plt.figure(figsize=(8, 8))
        plt.scatter(self.c_tsne[:, 0], self.c_tsne[:, 1], c=self.plot_label, cmap='tab20')
    
    def filter_phones_and_words_without_minimal_pairs_iteratively(self,
                                                                  display: bool=False,
                                                                  save: bool=False,
                                                                  print_phones: bool=False,
                                                                  print_words: bool=False):
        '''
        This function is used to filter out the phones and words that do not have minimal pairs.
        '''
        num_phones_differences = 1
        num_qualified_phones = len(self.found_phonemes)
        idx_iteration = 1
        
        qualified_words = self.found_words
        qualified_phones = {phone for word in qualified_words for phone in word.split()}
        
        total_eliminated_phones = 0
        
        while num_phones_differences > 0:
            print("-------------------------------------")
            print(f"Start the {idx_iteration}th Iteration of Filtering")
            print("The phones don't have minimal pairs will be eliminated")
            print("The words with invalid phones will be eliminated")
            print("-------------------------------------")  
            
            print(f"CURRENT NUMBER OF PHONES: {num_qualified_phones}.")
            print(f"CURRENT NUMBER OF WORDS: {len(qualified_words)}.")
            
            # group the words by the same length
            words_group = defaultdict(list)
            for word in qualified_words:
                words_group[len(word.split())].append(word)
            # print("The Number of Different Length Group", len(words_group))
            
            # get the minimal pair
            minimal_pair_context = defaultdict(list)
            for length, words in words_group.items():
                if length == 1 or length == 2:
                    continue
                for i in range(length):
                    for word in words:
                        phones = word.split()
                        phoneme_candidate = phones[i]
                        phones[i] = "*"
                        context = " ".join(phones)
                        minimal_pair_context[context].append(phoneme_candidate)
            
            # the last iteration makes sure that we have filtered out 
            # all the phones that don't have minimal pairs
            self.minimal_pairs = minimal_pair_context
                        
            # calculate prior probability of each phoneme
            # ========= Rethink how to measure the importance of a minimal pair =========
            # phoneme_prob = defaultdict(float)
            # phoneme_context_prob = defaultdict(list)
            # for context, phonemes in minimal_pair_context.items():
            #     if len(phonemes) > 1:
            #         prob_score = 0
            #         for phoneme in phonemes:
            #             prob_score += self.all_word_prior_prob.get(context.replace("*", phoneme), 0)
            #         prob_score /= len(phonemes)
                    
            #         for phoneme in phonemes:
            #             phoneme_context_prob[phoneme].append((context, prob_score))
            #             phoneme_prob[phoneme] += prob_score
            
            # get new qualified phones
            new_qualified_phones = set()
            for context, phonemes in minimal_pair_context.items():
                if len(phonemes) > 1:
                    new_qualified_phones.update(phonemes)
            # get new qualified words
            new_qualified_words = {word: cnt for word, cnt in qualified_words.items() if all(phoneme in new_qualified_phones for phoneme in word.split())}
            
            # get differences bewteen the new qualified phones and the old qualified phones
            num_phones_differences = num_qualified_phones - len(new_qualified_phones)
            num_words_differences = len(qualified_words) - len(new_qualified_words)
            
            total_eliminated_phones += num_phones_differences
            
            print(f"{num_phones_differences} phones and {num_words_differences} words are eliminated.")
            
            # prepare for the next iteration
            num_qualified_phones = len(new_qualified_phones)
            qualified_words = new_qualified_words
            qualified_phones = new_qualified_phones
            idx_iteration += 1
        
        if save:
            # save minimal pair context
            minimal_pair_context_filename = os.path.join(self.CK_PATH, "minimal_pair_context_phone.json")
            with open(minimal_pair_context_filename, 'w') as f:
                json.dump(minimal_pair_context, f)
        
        print("-------------------------------------")
        print("The Final Result of Filtering")
        print("-------------------------------------")
        print("We have eliminated {} phones in total.".format(total_eliminated_phones))
        print(f"We have {num_qualified_phones} qualified phones.")
        print(f"We have {len(qualified_words)} qualified words.")
        print()
        
        # print the final qualified phones and words
        if print_phones:
            for idx, phone in enumerate(sorted(list(qualified_phones))):
                if idx % 5 == 0 and idx != 0:
                    print()
                print(phone, end='\t')
        if print_words:
            for idx, word in enumerate(sorted(list(qualified_words.keys()))):
                if idx % 5 == 0 and idx != 0:
                    print()
                print(word, end='\t')
        
        self.found_words = qualified_words
        self.found_phonemes = qualified_phones
        
        return
    
    def find_context_of_each_word_in_a_sentence(self, sentence: list[str], context_mode: str, **kwargs):
        '''
        This function is used to find the context of each word in a sentence.
        It can apply on both grammar and energy parsing.
        And it can find the context of each word in terms of phoneme, word, and ngram.
        
        Input:
            sentence: list, the list of words in a sentence
            parse_mode: str, "grammar" or "energy"
            context_mode: str, "phoneme", "word", or "ngram"
            **kwargs: "num_context_phoneme" for phoneme context, "num_ngram" for ngram context
            
        Output:
            word_context: defaultdict(defaultdict(int)), the context of each word
        '''
        word_context = defaultdict(lambda: defaultdict(int))
        phone_sequence = list()
        
        # Get phone sequence
        if self.parse_mode == "grammar":
            for word in sentence:
                phone_sequence.extend(word.split())
        elif self.parse_mode == "energy":
            for word in sentence:
                if word.startswith("w"):
                    phone_sequence.extend(word[1:].split())
                else:
                    phone_sequence.append(word)
                    
        # Get start and end idx of each word
        word_idx = list()
        if self.parse_mode == "grammar":
            current_idx = 0
            for word in sentence:
                start_idx = current_idx
                end_idx = start_idx + len(word.split()) - 1
                word_idx.append((start_idx, end_idx))
                current_idx = end_idx + 1
        elif self.parse_mode == "energy":
            current_idx = 0
            for word in sentence:
                if word.startswith("w"):
                    start_idx = current_idx
                    end_idx = start_idx + len(word[1:].split()) - 1
                    word_idx.append((start_idx, end_idx))
                    current_idx = end_idx + 1
                else:
                    current_idx += 1
        
        # Get the context
        if self.parse_mode == "grammar": 
            if context_mode == "phoneme":
                num_context_phoneme: int = kwargs["num_context_phoneme"]
                for idx, word in enumerate(sentence):
                    start_idx, end_idx = word_idx[idx]
                    for i in range(max(0, start_idx - num_context_phoneme), min(len(phone_sequence), end_idx + num_context_phoneme + 1)):
                        if i < start_idx or i > end_idx:
                            word_context[word][phone_sequence[i]] += 1
            
            elif context_mode == "word":
                for idx, word in enumerate(sentence):
                    if idx - 1 >= 0:
                        word_context[word][sentence[idx - 1]] += 1
                    else:
                        word_context[word]["begin"] += 1
                    if idx + 1 < len(sentence):
                        word_context[word][sentence[idx + 1]] += 1
                    else:
                        word_context[word]["end"] += 1
            
            elif context_mode == "ngram":
                num_ngram: int = kwargs["num_ngram"]
                for idx, word in enumerate(sentence):
                    start_idx, end_idx = word_idx[idx]
                    previous_ngram = " ".join(phone_sequence[max(0, start_idx - num_ngram): start_idx])
                    following_ngram = " ".join(phone_sequence[end_idx + 1: min(len(phone_sequence), end_idx + num_ngram + 1)])
                    if previous_ngram:
                        word_context[word][previous_ngram] += 1
                    if following_ngram:
                        word_context[word][following_ngram] += 1
                    
        elif self.parse_mode == "energy":
            # get word list
            word_sequence = [word[1:] for word in sentence if word.startswith("w")]
            
            if context_mode == "phoneme":
                num_context_phoneme: int = kwargs["num_context_phoneme"]
                for idx, word in enumerate(word_sequence):
                    start_idx, end_idx = word_idx[idx]
                    for i in range(max(0, start_idx - num_context_phoneme), min(len(phone_sequence), end_idx + num_context_phoneme + 1)):
                        if i < start_idx or i > end_idx:
                            word_context[word][phone_sequence[i]] += 1
                        elif i == 0:
                            word_context[word]["begin"] += 1
                        elif i == len(phone_sequence) - 1:
                            word_context[word]["end"] += 1
            
            elif context_mode == "word":
                for idx, word in enumerate(word_sequence):
                    if idx - 1 >= 0:
                        word_context[word][word_sequence[idx - 1]] += 1
                    else:
                        word_context[word]["begin"] += 1
                    
                    if idx + 1 < len(word_sequence):
                        word_context[word][word_sequence[idx + 1]] += 1
                    else:
                        word_context[word]["end"] += 1
            
            elif context_mode == "ngram":
                num_ngram: int = kwargs["num_ngram"]
                for idx, word in enumerate(word_sequence):
                    start_idx, end_idx = word_idx[idx]
                    previous_ngram = " ".join(phone_sequence[max(0, start_idx - num_ngram): start_idx])
                    following_ngram = " ".join(phone_sequence[end_idx + 1: min(len(phone_sequence), end_idx + num_ngram + 1)])
                    if previous_ngram:
                        word_context[word][previous_ngram] += 1
                    else:
                        word_context[word]["begin"] += 1
                    if following_ngram:
                        word_context[word][following_ngram] += 1
                    else:
                        word_context[word]["end"] += 1
                    
        return word_context
    
    def find_context_of_words(self, context_mode: str, **kwargs):
        
        word_context = defaultdict(lambda: defaultdict(int))
        
        # walk through the transcriptions and get all contexts
        for transcription in self.transcriptions:
            word_context_in_sentence = self.find_context_of_each_word_in_a_sentence(transcription, context_mode, **kwargs)
            for k, v in word_context_in_sentence.items():
                for inner_k, inner_v in v.items():
                    word_context[k][inner_k] += inner_v
                    
        # filter the word list
        word_context = {word: context for word, context in word_context.items() if word in list(self.found_words.keys())}
        
        assert len(word_context) == len(self.found_words), \
            f"The number of words in word context: {len(word_context)} \
                is not equal to the number of words in found words: {len(self.found_words)}."
        
        # get all minimal pairs with phoneme in the same context
        # all_minimal_paris_with_phooneme = [(context, (phoneme1, phoneme2)), ...]
        all_minimal_pairs_with_phoneme = list()
        for context, phonemes in self.minimal_pairs.items():
            pairs_with_same_environment = list(combinations(phonemes, 2))
            for pair in pairs_with_same_environment:
                all_minimal_pairs_with_phoneme.append((context, tuple(sorted(pair))))
                
        # calculate the KL divergence
        phoneme_pair_context_kl_div = defaultdict(lambda: defaultdict(float))
        
        for context, pair in all_minimal_pairs_with_phoneme:
            word1, word2 = context.replace("*", pair[0]), context.replace("*", pair[1])
            word1_context, word2_context = word_context[word1], word_context[word2]
            kl_div = calculate_js_divergence(word1_context, word2_context)
            
            phoneme_pair_context_kl_div[pair][context] = kl_div
            
        # sort phoneme pair context by the sum of kl divergence
        phoneme_pair_context_kl_div = dict(sorted(phoneme_pair_context_kl_div.items(), key=lambda x: sum(v for k, v in x[1].items()), reverse=True))
        # sort the kl divergence of each phoneme pair
        for pair, context_kl in phoneme_pair_context_kl_div.items():
            phoneme_pair_context_kl_div[pair] = dict(sorted(context_kl.items(), key=lambda x: x[1], reverse=True))
        
        kl_div_list = list()
        for pair, context_kl in phoneme_pair_context_kl_div.items():
            kl_div_list.extend(list(context_kl.values()))
        
        # get the mean and mediean and 25% percentile of kl divergence
        dl_div_mean = np.mean(kl_div_list)
        kl_div_median = np.median(kl_div_list)
        kl_div_95_percentile = np.percentile(kl_div_list, 95)
        kl_div_75_percentile = np.percentile(kl_div_list, 75)
        kl_div_50_percentile = np.percentile(kl_div_list, 50)
        kl_div_25_percentile = np.percentile(kl_div_list, 25)
        kl_div_5_percentile = np.percentile(kl_div_list, 5)
        
        self.percentile_custom = np.percentile(kl_div_list, self.percentile)
        self.percentile_95_list.append(kl_div_95_percentile)
        self.percentile_75_list.append(kl_div_75_percentile)
        self.percentile_50_list.append(kl_div_50_percentile)
        self.percentile_25_list.append(kl_div_25_percentile)
        self.percentile_5_list.append(kl_div_5_percentile)
        
        print("The Mean of KL Divergence: ", dl_div_mean)
        print("The Median of KL Divergence: ", kl_div_median)
        print("The 25% Percentile of KL Divergence: ", kl_div_25_percentile)
        print("The 5% Percentile of KL Divergence: ", kl_div_5_percentile)
        
        return phoneme_pair_context_kl_div
    
    def _update_mapping(self, replacement_mapping: dict):
        for ori, replace in self.mapping_phoneme.items():
            if replace in replacement_mapping.keys():
                self.mapping_phoneme[ori] = replacement_mapping[replace]
        self.mapping_phoneme.update(replacement_mapping)
    
    def update_phonemes_words(self, phoneme_pair_context_kl_div: dict):
        replacement_mapping = dict()
        
        # if we don't set a threshold, we will use the 5% percentile of kl divergence
        if not self.percentile:
            phoneme_merge_threshold = self.percentile_5_list[-1]
        else:
            phoneme_merge_threshold = self.percentile_custom
        
        for pair, context_kl in phoneme_pair_context_kl_div.items():
            if all(kl_div < phoneme_merge_threshold for kl_div in context_kl.values()):
                min_phone = min(int(pair[0]), int(pair[1]))
                max_phone = max(int(pair[0]), int(pair[1]))
                replacement_mapping[str(max_phone)] = str(min_phone)
                
        # sort the replacement mapping
        replacement_mapping = dict(sorted(replacement_mapping.items(), key=lambda x: int(x[1]), reverse=True))
        self.this_round_mapping_phoneme = replacement_mapping
        self.mapping_phoneme.update(replacement_mapping)
        print("This round will replace {} phonemes".format(len(replacement_mapping))) 
        print(replacement_mapping)
        
        # update the transcriptions
        if self.parse_mode == "energy":
            # Split transcriptions into chunks
            num_cores = mp.cpu_count()
            chunk_size = len(self.transcriptions) // num_cores
            transcription_chunks = [self.transcriptions[i:i + chunk_size] for i in range(0, len(self.transcriptions), chunk_size)]
            
            # Create a pool of workers and process the chunks in parallel
            with mp.Pool(processes=num_cores) as pool:
                results = pool.starmap(process_transcription_chunk, [(chunk, replacement_mapping) for chunk in transcription_chunks])

            # Merge the results from all processes
            self.transcriptions = [transcription for chunk in results for transcription in chunk]
            
            transcription_chunks = [self.raw_transcriptions[i:i + chunk_size] for i in range(0, len(self.raw_transcriptions), chunk_size)]
            with mp.Pool(processes=num_cores) as pool:
                results = pool.starmap(process_transcription_chunk, [(chunk, replacement_mapping) for chunk in transcription_chunks])
            self.raw_transcriptions = [transcription for chunk in results for transcription in chunk]
        
        elif self.parse_mode == "grammar":
            for idx_sentence, transcription in enumerate(self.transcriptions):
                for idx_word, word in enumerate(transcription):
                    phones = word.split()
                    for i in range(len(phones)):
                        if phones[i] in replacement_mapping.keys():
                            phones[i] = replacement_mapping[phones[i]]
                    new_word = " ".join(phones)
                    transcription[idx_word] = new_word
        
        phoneme = set()
        for _, transcription in enumerate(self.transcriptions):
            for _, word in enumerate(transcription):
                if word.startswith("w"):
                    for phone in word[1:].split():
                        phoneme.add(phone)
         
        # merge identical phones in a word
        if self.parse_mode == "energy":
            # Split transcriptions into chunks
            num_cores = mp.cpu_count()
            chunk_size = len(self.transcriptions) // num_cores
            transcription_chunks = [self.transcriptions[i:i + chunk_size] for i in range(0, len(self.transcriptions), chunk_size)]
            
            # Create a pool of workers and process the chunks in parallel
            with mp.Pool(processes=num_cores) as pool:
                results = pool.map(process_transcription_deduplication, transcription_chunks)

            # Merge the results from all processes
            self.transcriptions = [transcription for chunk in results for transcription in chunk]
            
            # Merge the raw transcriptions
            self.single_phone_transcriptions = []
            for idx_sentence, transcription in enumerate(self.transcriptions):
                sp_trans = []
                for word in transcription:
                    if word.startswith("w"):
                        sp_trans.extend(word[1:].split())
                    else:
                        sp_trans.append(word)
                self.single_phone_transcriptions.append(sp_trans.copy())
            
            
                
            # assert len(self.raw_transcriptions) == len(self.phone_ts), \
            #     f"The number of raw transcriptions: {len(self.raw_transcriptions)} \
            #         is not equal to the number of phone timestamps: {len(self.phone_ts)}."
            # for idx, (raw_transcription, phone_timestamp) in enumerate(zip(self.raw_transcriptions, self.phone_ts)):
            #     if len(raw_transcription) < 1:
            #         continue
            #     phone = raw_transcription[0]
            #     new_raw_transcription = [phone]
            #     assert len(phone_timestamp) == len(raw_transcription), \
            #         f"The number of phone timestamps: {len(phone_timestamp)} \
            #             is not equal to the number of phones in raw transcription: {len(raw_transcription)}."
            #     new_phone_timestamp = [[phone_timestamp[0][0], phone_timestamp[0][1]]]
            #     for i in range(1, len(raw_transcription)):
            #         if raw_transcription[i] != raw_transcription[i - 1]:
            #             phone = raw_transcription[i]
            #             new_raw_transcription.append(phone)
            #             new_phone_timestamp.append([phone_timestamp[i][0], phone_timestamp[i][1]])
            #         else:
            #             self.phone_in_word[idx][i] = 2
            #             new_phone_timestamp[-1][1] = phone_timestamp[i][1]
                # self.phone_in_word[idx] = [phone_in_word for phone_in_word in self.phone_in_word[idx] if phone_in_word != 2]
                # assert len(self.phone_in_word[idx]) == len(new_raw_transcription), \
                #     f"The number of phones in word: {len(self.phone_in_word[idx])} \
                #         is not equal to the number of phones in raw transcription: {len(new_raw_transcription)}."
                
                # self.raw_transcriptions[idx] = new_raw_transcription
                # self.phone_ts[idx] = new_phone_timestamp
            
                
        # update the found phonemes and words
        self.found_words = defaultdict(int)
        self.found_phonemes = set()
        
        if self.parse_mode == "energy":
            for transcription in self.transcriptions:
                for word in transcription:
                    if word.startswith("w"):
                        self.found_words[word[1:]] += 1
        
        elif self.parse_mode == "grammar":
            for transcription in self.transcriptions:
                for word in transcription:
                    self.found_words[word] += 1
        
        if self.eliminate_noise_manually: 
            # eliminate the words containing noise phones
            self.found_words = {word: cnt for word, cnt in self.found_words.items() if not any([int(phone) in self.NOISE_PHONES for phone in word.split()])}
        
        # eliminate the words cnt less than threshold
        self.found_words = {word: cnt for word, cnt in self.found_words.items() if cnt > self.valid_word_cnt_threshold}
        
        # update the found phonemes
        if self.parse_mode == "energy":
            self.found_phonemes = {phone for word in self.found_words for phone in word.split()}
        elif self.parse_mode == "grammar":
            self.found_phonemes = {phone for word in self.found_words for phone in word.split()}
        
    def plot_cluster_center_feature(self, idx_print_phonemes: list, parse_mode: str, context_mode: str, round_idx: int):
        
        c_tsne = self.c_tsne
        label = self.plot_label
        
        # plot the t-SNE in 2D of the idx_print_phonemes
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_xlim(-10, 11)
        ax.set_ylim(-10, 9)
        # plt.scatter(c_tsne[:, 0], c_tsne[:, 1], c=label, cmap='tab20')
        
        new_label = [str(i) for i in idx_print_phonemes]
        for i, txt in enumerate(new_label):
            plt.scatter(c_tsne[idx_print_phonemes[i], 0], c_tsne[idx_print_phonemes[i], 1], color='blue')
            plt.annotate(txt, (c_tsne[idx_print_phonemes[i], 0], c_tsne[idx_print_phonemes[i], 1]), fontsize=10, ha='center')
            
        be_merged_phonemes = [int(phone) for phone in self.this_round_mapping_phoneme.keys()]
        
        # plot X on the phonemes that will be merged
        for i, txt in enumerate(be_merged_phonemes):
            plt.scatter(c_tsne[txt, 0], c_tsne[txt, 1], marker='x', color='red')
            plt.annotate(txt, (c_tsne[txt, 0], c_tsne[txt, 1]), fontsize=10, ha='center')
        
        plt.title(f"t-SNE of the Cluster Centers - Round {round_idx}")
        plt.legend()
        
        is_noise = "denoise" if self.eliminate_noise_manually else "noise"
        img_parent_path = os.path.join("saved_imgs", parse_mode + "_" + context_mode + "_" + is_noise + "_persentaile_" + str(self.percentile))
        os.makedirs(img_parent_path, exist_ok=True)
        plt.savefig(os.path.join(img_parent_path, f"t-SNE_cluster_centers_hl_{parse_mode}_{context_mode}_{is_noise}_{round_idx}.png"))
    
    def find_phonemes(self, context_mode: str, num_context_phoneme: int=3, phoneme_merge_threshold: float=0.1):
        '''
        This function is used to find the phonemes by using minimal pairs.
        Step 1: Find the minimal pairs and the word context of each pair.
        Step 2: Calculate the KL divergence between each pair of phonemes.
        Step 3: Merge the phonemes if the JS divergence is smaller than the threshold.
        '''
        print("Start Finding Phonemes...")
        print()
        self.found_words = {word: cnt for word, cnt in self.all_words_cnt.items() if word in self.valid_word_list}
        self.found_phonemes = {phone for word in self.found_words for phone in word.split()}
        
        previous_num_phonemes = 0
        current_num_phonemes = len(self.found_phonemes)
        idx_round = 1
        
        while current_num_phonemes != previous_num_phonemes:
            self.filter_phones_and_words_without_minimal_pairs_iteratively(display=False)
            phoneme_context_pair_kl_div = self.find_context_of_words(context_mode, num_context_phoneme=num_context_phoneme)
            self.update_phonemes_words(phoneme_context_pair_kl_div)
            
            previous_num_phonemes = current_num_phonemes
            current_num_phonemes = len(self.found_phonemes)
            
            word_dict = {word: cnt for word, cnt in self.found_words.items()}
            self.get_word_list_and_word_cnt_dict()
            print(f"After {idx_round}th Round of Phoneme Finding, \
                The Number of Phonemes in words which appear over \
                    {self.valid_word_cnt_threshold} times is {current_num_phonemes}.")
            
            # self.plot_cluster_center_feature([int(phoneme) for phoneme in self.found_phonemes], self.parse_mode, context_mode, idx_round)

            idx_round += 1
        
        # save kl divergence
        # str the kl divergence
        str_phoneme_context_pair_kl_div = {str(pair): {context: kl_div for context, kl_div in context_kl.items()} for pair, context_kl in phoneme_context_pair_kl_div.items()}
        kl_file_path = os.path.join(self.CK_PATH, "kl_divergence_word.json")
        mapping_file_path = os.path.join(self.CK_PATH, "mapping_phoneme.json")
        with open(mapping_file_path, 'w') as f:
            json.dump(self.mapping_phoneme, f)
        with open(kl_file_path, 'w') as f:
            json.dump(str_phoneme_context_pair_kl_div, f)
        self.save_config()
        
        self.filter_phones_and_words_without_minimal_pairs_iteratively(display=True)
        print("Phoneme Finding Finished.")
        print("Total {} phones has been merged.".format(len(self.mapping_phoneme)))
        print("The Found Phonemes: ", self.found_phonemes)
        print("The Found Words: ")
        for idx, (word, cnt) in enumerate(self.found_words.items()):
            if idx == 10:
                break
            print(word, cnt)
        print("...")
        
    def generate_phoneme_comparison_samples(self):
        # refine phoneme replacement mapping
        self.merged_phoneme_sets_list = list(merge_sets([set([int(k), int(v)]) for k, v in self.mapping_phoneme.items()]))
        
        print("The Number of Merged Phoneme Sets: ", len(self.merged_phoneme_sets_list))
        print("The Merged Phoneme Sets: ", self.merged_phoneme_sets_list)
        
        # prepare evaluation samples
        all_phone_samples_parent_path = "phone_samples"
        phone_samples_paths = dict()
        for phone_name in os.listdir(all_phone_samples_parent_path):
            phone_samples_paths[phone_name] = list()
            for phone_sample in os.listdir(os.path.join(all_phone_samples_parent_path, phone_name)):
                phone_samples_paths[phone_name].append(os.path.join(all_phone_samples_parent_path, phone_name, phone_sample))
        
        # merge phone samples if they are in the same set
        for replacement_set in self.merged_phoneme_sets_list:
            replace_with = str(min(replacement_set))
            for phone in replacement_set:
                if phone != int(replace_with):
                    phone_samples_paths[replace_with].extend(phone_samples_paths[str(phone)])
                    del phone_samples_paths[str(phone)]
        
        same_phone_pairs = list()
        diff_phone_pairs = list()
        # get same phone pairs
        for phone_name in self.found_phonemes:
            # randomly select 10 samples for each phone
            if len(phone_samples_paths[phone_name]) < 10:
                raise ValueError("The number of samples for each phone should be larger than 10")
            selected_samples = random.sample(phone_samples_paths[phone_name], 10)
            # shuffle the selected samples
            random.shuffle(selected_samples)
            for i in range(5):
                same_phone_pairs.append((selected_samples[i], selected_samples[i + 5]))
        
        # get diff phone pairs
        diff_phone_candidates = dict()
        for phone_name in self.found_phonemes:
            diff_phone_candidates[phone_name] = random.sample(phone_samples_paths[phone_name], 10)
        
        all_files = [(phone, file) for phone, files in diff_phone_candidates.items() for file in files]
        random.shuffle(all_files)
        used_files = set()
        
        while len(used_files) < len(all_files):
            # Find a pair with files from different phones
            for i in range(len(all_files)):
                if all_files[i][1] in used_files:
                    continue
                for j in range(i + 1, len(all_files)):
                    if all_files[j][1] in used_files:
                        continue
                    phone1, file1 = all_files[i]
                    phone2, file2 = all_files[j]
                    if phone1 != phone2:
                        diff_phone_pairs.append((file1, file2))
                        used_files.add(file1)
                        used_files.add(file2)
                        break
                if len(used_files) % 2 == 0:  # Check after inner loop if pair is found
                    break
        
        assert len(same_phone_pairs) == len(diff_phone_pairs), f"The number of same phone pairs is {len(same_phone_pairs)}, the number of different phone pairs is {len(diff_phone_pairs)}"
        
        # save the samples
        sample_folder_path = os.path.join("merge_phone_eval_samples_c5_p15", "samples")
        os.makedirs(sample_folder_path, exist_ok=True)
        
        all_samples_pairs_with_answer = list()
        for same_pair in same_phone_pairs:
            all_samples_pairs_with_answer.append((same_pair[0], same_pair[1], 1))
        for diff_pair in diff_phone_pairs:
            all_samples_pairs_with_answer.append((diff_pair[0], diff_pair[1], 0))
        random.shuffle(all_samples_pairs_with_answer)
        
        answer_sheet = list()
        for idx, pair in enumerate(all_samples_pairs_with_answer):
            pair_folder_path = os.path.join(sample_folder_path, str(idx))
            os.makedirs(pair_folder_path, exist_ok=True)
            
            shutil.copy(pair[0], os.path.join(pair_folder_path, "0.wav"))
            shutil.copy(pair[1], os.path.join(pair_folder_path, "1.wav"))
            answer_sheet.append((pair[2]))
        
        # save as json file
        with open(os.path.join("merge_phone_eval_samples_c5_p15", "answer_sheet.json"), 'w') as f:
            json.dump(answer_sheet, f)
            
    def get_word_timestamps_in_transcription(self):
        audio_filename_file = "short-dog/train_audio.dat"
        audio_folder = "/bigdrive/dog/wav_audios"
        input_folder = "short-dog/24Jun14-222049-D31294-P10-S10-B128-O1000-t512-k0.8-Ggrammar.unigram"
        ori_path = "short-dog/train_ori.dat"
        
        self.word_video_cut_dict = dict()
        
        # load data
        for file in os.listdir(input_folder):
            if file.endswith(".test"):
                with open(os.path.join(input_folder, file), "r") as f:
                    prediction_lines = f.readlines()
            elif file.endswith(".truth"):
                with open(os.path.join(input_folder, file), "r") as f:
                    truth_lines = f.readlines()

        with open(ori_path, "r") as f:
            ori_lines = f.readlines()
        
        with open(audio_filename_file, "r") as f:
            audio_lines = f.readlines()
        
        assert len(prediction_lines) == len(truth_lines) == len(ori_lines) == len(audio_lines) == len(self.transcriptions), "The number of lines should be the same"
        dictionary_log = [["word", "filename", "word_path", "start_time", "end_time", "idx_sentence", "idx_word"]]
        
        for iddx, transcription in enumerate(self.transcriptions):
            # get duration of each phone first
            audio_filename = audio_lines[iddx].strip()
            audio_path = os.path.join(audio_folder, audio_filename)
            ori: list = ori_lines[iddx].strip().split()
            truth: list = truth_lines[iddx].strip().split()
            
            whole_duration = librosa.get_duration(path=audio_path)
            duration_per_phone = whole_duration / len(ori)
            timestamps_for_combined_phonemes = [0]
            combined_sen = []
            for idx, phoneme in enumerate(ori):
                if len(combined_sen) < 1:
                    combined_sen.append(phoneme)
                else:
                    if phoneme == combined_sen[-1]:
                        continue
                    else:
                        combined_sen.append(phoneme)
                        timestamps_for_combined_phonemes.append(idx)
            timestamps_for_combined_phonemes.append(len(ori))
            assert len(timestamps_for_combined_phonemes) == len(truth) + 1, "The number of timestamps should be the same as the number of words"
            
            num_previous_phone = 0
            if self.parse_mode == "energy":
                for idx, word in enumerate(transcription):
                    if word.startswith("w"):
                        word = word[1:]
                        start_time = timestamps_for_combined_phonemes[num_previous_phone] * duration_per_phone
                        number = len(word.split())
                        end_time = timestamps_for_combined_phonemes[num_previous_phone + len(word.split())] * duration_per_phone
                        dictionary_log.append([word, audio_filename, audio_path, max(0, start_time-2), min(whole_duration, end_time+2), iddx, idx])
                        self.word_video_cut_dict[(iddx, idx)] = {"filename": audio_filename, "filepath": audio_path, "start_time": max(0, start_time-2), "end_time": min(whole_duration, end_time+2)}
                    num_previous_phone += len(word.split())
            elif self.parse_mode == "grammar":
                for idx, word in enumerate(transcription):
                    start_time = timestamps_for_combined_phonemes[num_previous_phone] * duration_per_phone
                    end_time = timestamps_for_combined_phonemes[num_previous_phone + len(word.split())] * duration_per_phone
                    dictionary_log.append([word, audio_filename, audio_path, max(0, start_time-2), min(whole_duration, end_time+2), iddx, idx])
                    self.word_video_cut_dict[[iddx, idx]] = {"filename": audio_filename, "filepath": audio_path, "start_time": max(0, start_time-2), "end_time": min(whole_duration, end_time+2)}
                    num_previous_phone += len(word.split())
        
    def generate_word_video_cut_samples(self, word_list: list):
        audio_filename_file = "short-dog/train_audio.dat"
        audio_parent_folder = "/bigdrive/dog/wav_audios"
        with open(audio_filename_file, "r") as f:
            audio_lines = f.readlines()
        
        word_timestamps_file = "energy_word_timestamps.json"
        with open(word_timestamps_file, "r") as f:
            word_timestamps = json.load(f)
        
        all_info = list()
        # go through transcriptions
        if self.parse_mode == "energy":
            for idx_sentence, transcription in enumerate(self.transcriptions):
                audio_file_path = os.path.join(audio_parent_folder, audio_lines[idx_sentence].strip())
                word_idx=0
                for idx_word, word in enumerate(transcription):
                    if word.startswith("w"):
                        if word[1:] in word_list:
                            start_time, end_time = word_timestamps[idx_sentence][word_idx]
                            info = {"word": word[1:], "filename": audio_lines[idx_sentence], "filepath": audio_file_path, "start_time": start_time, "end_time": end_time}
                            all_info.append(info)
                        word_idx += 1
        # save the dataframe
        
        df = pd.DataFrame(all_info)
        df.to_csv("word_video_cut_samples.csv", index=False)

    def generate_phone_samples(self, duration_threshold: float=0.1, num_samples: int=10):
        assert self.parse_mode == "energy", "This function is only for energy parsing."
        assert len(self.phone_ts) == len(self.transcriptions), \
            f"The number of phone timestamps: {len(self.phone_ts)} \
                is not equal to the number of transcriptions: {len(self.transcriptions)}."
        
        '''
        self.found_phone_samples = {phone: [{filepath: "",
                                            start_time: 0.0,
                                            end_time: 0.0}, {}, ...],
                                    ...}
        '''
        self.found_phone_samples = defaultdict(list)
        
        for idx, (ts, transcription) in enumerate(zip(self.phone_ts, self.transcriptions)):
            # get pure phone sequence
            phone_list = list()
            for word in transcription:
                if word.startswith("w"):
                    phone_list.extend(word[1:].split())
                else:
                    phone_list.append(word)
            assert len(phone_list) == len(ts), \
                f"The number of phones in transcription: {len(phone_list)} \
                    is not equal to the number of timestamps: {len(ts)}."
            
            for i in range(len(phone_list)):
                if (ts[i][1] - ts[i][0]) > duration_threshold:
                    if phone_list[i] in self.found_phonemes:
                        audio_path = self.audio_paths[idx]
                        start_time = ts[i][0]
                        end_time = ts[i][1]
                        self.found_phone_samples[phone_list[i]].append(
                            {"filepath": audio_path,
                             "start_time": start_time,
                             "end_time": end_time}
                        )
        
        # randomly select num_samples samples for each phone
        for phone, samples in self.found_phone_samples.items():
            save_folder = os.path.join(self.CK_PATH, "phone_samples", phone)
            os.makedirs(save_folder, exist_ok=True)
            
            if len(samples) < num_samples:
                new_num_samples = len(samples)
            else:
                new_num_samples = num_samples
                
            selected_samples = random.sample(samples, new_num_samples)
            
            for idx, sample in enumerate(selected_samples):
                audio_path = sample["filepath"]
                start_time = sample["start_time"]
                end_time = sample["end_time"]
                filename = audio_path.split("/")[-1].split(".")[0]
                str_start_time = f"{start_time:.2f}".replace(".", "_")
                str_end_time = f"{end_time:.2f}".replace(".", "_")
                save_path = os.path.join(save_folder, f"{filename}_{str_start_time}_{str_end_time}.wav")
                cut_audio(audio_path, start_time, end_time, save_path)
                        
            
    
if __name__ == "__main__":

    # # Test the phoneme finder
    metadata_file = "/data/xingyuan/animal_language/v2/out/fairseq/six_balance/work/iter2_130k_11th_km250/metadata_trans.json"
    # metadata_file = "/data/xingyuan/animal_language/v2/out/fairseq/six_balance/work/iter2_130k_11th_km500/metadata_trans.json"
    phoneme_finder = PhonemeFinder(parse_mode="energy", parsed_transcription_file_path=metadata_file, valid_word_cnt_threshold=5, eliminate_noise_manually=False, percentile=5)
    # # phoneme_finder.get_word_timestamps_in_transcription()
    # phoneme_finder.plot_cluster_center_feature([int(phoneme) for phoneme in phoneme_finder.found_phonemes], phoneme_finder.parse_mode, context_mode="phoneme", round_idx=0)
    phoneme_finder.find_phonemes(context_mode="phoneme", num_context_phoneme=3, phoneme_merge_threshold=0.0)
    # phoneme_finder.generate_phone_samples()
    # word_list = [word for word, cnt in phoneme_finder.found_words.items() if cnt > 100]
    # total_num_words = sum(phoneme_finder.found_words.values())
    # print("The Number of Words: ", len(word_list))
    # print(list(merge_sets([set([int(k), int(v)]) for k, v in phoneme_finder.mapping_phoneme.items()])))
    
    # for phoneme in phoneme_finder.found_phonemes:
    #     print(phoneme, end=" ")
    
    # # sort the found words by the number of appearance
    # found_words = dict(sorted(phoneme_finder.found_words.items(), key=lambda x: x[1], reverse=True))
    
    # print()
    # items = list(found_words.items())
    # for i in range(0, len(items), 2):
    #     word1, cnt1 = items[i]
    #     if i + 1 < len(items):
    #         word2, cnt2 = items[i + 1]
    #         print(f"\\verb|{word1}| & {cnt1} & \\verb|{word2}| & {cnt2} \\\\")

        
    
    # phoneme_finder.generate_phoneme_comparison_samples()
    # phoneme_finder.generate_word_video_cut_samples(word_list)
    
    