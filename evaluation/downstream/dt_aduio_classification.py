import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import datetime
import json
import torch
import librosa
import numpy as np
import pandas as pd
from joblib import load
from collections import Counter
from transformers import Wav2Vec2Tokenizer
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from tqdm import tqdm
from scipy.ndimage import zoom
from models import BERTForAudioClassification, BERTForPhonemeClassification
from sklearn.metrics import f1_score, accuracy_score, hamming_loss
from torch.optim.lr_scheduler import ReduceLROnPlateau

from constant import get_mfcc_fea, get_mel_spectro_fea, get_gemaps_fea, get_egemaps_fea

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report
from sklearn.multiclass import OneVsRestClassifier


class DogBarkTypeDataset(Dataset):
    def __init__(self, audios_folder_path: str,
                 label_file_path: str,):
        
        self.max_length = 1000
        
        print("start loading data and store in memory")
        self.audio_filenames = list(os.listdir(audios_folder_path))
        # read csv file
        filepath_label_dict = {}
        with open(label_file_path, 'r') as f:
            for line in f:
                filepath, label = line.strip().split(',')[:2]
                filepath_label_dict[filepath] = label
        
        self.labels = []
        self.audios = []
        for audio_file in self.audio_filenames:
            self.labels.append(filepath_label_dict[audio_file].split())
            audio_path = os.path.join(audios_folder_path, audio_file)
            features = self._process_audio(audio_path) 
            # exchange dimension of features
            features = features.transpose(1, 0)
            self.audios.append(features)
        self.labels = self._quantize_label(self.labels)
        print("finish loading data and store in memory")
        n = 0
        for label in self.labels:
            if sum(label) == 1:
                n += 1
        print(f"Number of single label: {n}")
    
    def _load_and_pad_audio(self,
                            audio_path: str,
                            max_length: int):
        audio, sr = librosa.load(audio_path)
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
        
        if mfccs.shape[1] > max_length:
            mfccs = zoom(mfccs, (1, max_length / mfccs.shape[1]))
        else:
            pad_width = max_length - mfccs.shape[1]
            mfccs = np.pad(mfccs, ((0, 0), (0, pad_width)), mode='constant')
            
        return mfccs, sr
    
    def _process_audio(self, audio_path):
        audio_padded, sr = self._load_and_pad_audio(audio_path, self.max_length)
        return audio_padded
    
    def _quantize_label(self, labels: list):
        """
        Quantize the labels to multi-labels
        """
        # Quantize the labels to multi-labels
        unique_labels = {label for multi_labels in labels 
                         for label in multi_labels}
        unique_labels = sorted(list(unique_labels))
        label_dict = {label: i for i, label in enumerate(unique_labels)}
        self.num_labels = len(unique_labels)
        
        for label, i in label_dict.items():
            print(f"The position of {label} is {i}")
        
        quantized_labels = []
        for multi_label in labels:
            quantized_label = np.zeros(len(unique_labels))
            for label in multi_label:
                quantized_label[label_dict[label]] = 1
            quantized_labels.append(quantized_label)
        return quantized_labels
             
    def __len__(self):
        return len(self.audio_filenames)

    def __getitem__(self, idx):
        # Load and extract features
        features = self.audios[idx]
        label = self.labels[idx]
        return features, label


class DogBarkTypeTranscriptionDataset(Dataset):
    def __init__(self, label_file: str, max_length=500, pad_value=0):
        self.max_length = max_length
        self.pad_value = pad_value

        # Load metadata and initialize storage
        metadata = pd.read_csv(label_file)
        self.transcriptions = []
        self.labels = []

        # Preprocess the data efficiently
        self._load_data(metadata)
        self.labels = self._quantize_labels(self.labels)

    def _load_data(self, metadata):
        """
        Load and preprocess transcription data.
        """
        n = 0
        for _, row in metadata.iterrows():
            if pd.notna(row["trans"]):
                transcription = np.load(row["trans"]).astype(int).tolist()
                transcription = self._merge_consecutive_phonemes(transcription)
                transcription = self._pad_or_truncate(transcription)
                self.transcriptions.append(transcription)

                # Parse multi-labels (comma-separated string)
                label = row["label"].split()
                if len(label) == 1:
                    n += 1
                self.labels.append(label)
        print(f"Number of single label: {n}")
                
    def _merge_consecutive_phonemes(self, transcription):
        """
        Merge consecutive phonemes to reduce the sequence length.
        """
        merged_transcription = []
        current_phoneme = None
        for phoneme in transcription:
            if phoneme == current_phoneme:
                continue
            merged_transcription.append(phoneme)
            current_phoneme = phoneme
        return merged_transcription

    def _pad_or_truncate(self, sequence):
        """
        Ensure all sequences are of the same length by padding or truncating.
        """
        if len(sequence) < self.max_length:
            return sequence + [self.pad_value] * (self.max_length - len(sequence))
        else:
            return sequence[:self.max_length]

    def _quantize_labels(self, labels: list):
        """
        Quantize the labels to multi-labels
        """
        # Quantize the labels to multi-labels
        unique_labels = {label for multi_labels in labels 
                         for label in multi_labels}
        unique_labels = sorted(list(unique_labels))
        label_dict = {label: i for i, label in enumerate(unique_labels)}
        self.num_labels = len(unique_labels)
        
        for label, i in label_dict.items():
            print(f"The position of {label} is {i}")
        
        quantized_labels = []
        for multi_label in labels:
            quantized_label = np.zeros(len(unique_labels))
            for label in multi_label:
                quantized_label[label_dict[label]] = 1
            quantized_labels.append(quantized_label)
        return quantized_labels

    def __len__(self):
        return len(self.transcriptions)

    def __getitem__(self, idx):
        trans = torch.tensor(self.transcriptions[idx], dtype=torch.long)
        label = self.labels[idx]
        return trans, label

    
class Logger():
    def __init__(self, cp_dir: str):
        self.checkpoints_dir = os.path.join("/home/sinong/ALP/AniLang/Evaluation/checkpoints",
                                            cp_dir)
        
        # Create the folder
        os.makedirs(self.checkpoints_dir, exist_ok=True)
        
        self.log_filename = os.path.join(self.checkpoints_dir, "log.txt")
        self.config_filename = os.path.join(self.checkpoints_dir, "config.json")
        self.log_file = open(self.log_filename, 'w')
    
    def write(self, message):
        self.log_file.write(message)
        
    def close(self):
        self.log_file.close()
        
    def record_loss(self, epoch, loss):
        self.write(f"Epoch {epoch} loss: {loss}\n")
        
    def record_eval(self, epoch, f1_micro, f1_macro, acc, hamming_acc):
        self.write(f"Epoch {epoch} F1_Micro: {f1_micro}, F1_Macro: {f1_macro}"
                   + f", Accuracy: {acc}, Hamming Acc: {hamming_acc}\n")
        
    def record_config(self, config):
        with open(self.config_filename, 'w') as f:
            json.dump(config, f)
    
    
def train_six_bark_type(dataloader: DataLoader,
                        model,
                        optimizer: torch.optim.Optimizer):
    for features, labels in dataloader:
        output = model(features)
        loss = torch.nn.BCELoss(output, labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()


def calculate_class_weights(dataset):
    # Assuming your dataset contains the labels for all classes in multi-label format
    label_counts = Counter()
    total_samples = 0
    
    for _, labels in dataset:
        total_samples += 1
        # Assuming labels is a numpy array with binary values (multi-labels)
        label_counts.update(labels)  # Sum across all labels for each sample
    
    # Compute class weights: total_samples / label occurrence count (inverse frequency)
    class_weights = {}
    for label, count in label_counts.items():
        if count == 0:
            class_weights[label] = np.inf  # Handle zero counts (if label never appears)
        else:
            class_weights[label] = total_samples / count
    
    # Convert to torch tensor and normalize the weights
    class_weights_tensor = torch.tensor(
        [class_weights.get(i, 1.0) for i in range(len(label_counts))], dtype=torch.float32
    )
    
    return class_weights_tensor

        
def bark_type_main():
    train_audio_path = "/data/dogs/sixbarktype_dataset/audios/train"
    eval_audio_path = "/data/dogs/sixbarktype_dataset/audios/eval"
    unbalance_audio_path = "/data/dogs/sixbarktype_dataset/audios/unbalanced"
    
    train_label_path = "/data/dogs/sixbarktype_dataset/labels/train.csv"
    eval_label_path = "/data/dogs/sixbarktype_dataset/labels/eval.csv"
    unbalance_label_path = "/data/dogs/sixbarktype_dataset/labels/unbalanced.csv"
    
    num_epochs = 1000
    feature_dim = 13  # For MFCCs with 13 coefficients
    d_model = 768  # Transformer model dimension
    num_heads = 12  # Number of attention heads
    d_ff = 768  # Feedforward hidden layer dimension
    num_layers = 6  # Number of transformer blocks
    num_classes = 6  # Number of classes for classification
    max_position_embeddings = 1000  # Max number of audio frames
    dropout = 0.1
    checkpoints_dir = f"barktype/{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    logger = Logger(checkpoints_dir)
    
    # Save the configuration
    config = {
        "num_epochs": num_epochs,
        "feature_dim": feature_dim,
        "d_model": d_model,
        "num_heads": num_heads,
        "d_ff": d_ff,
        "num_layers": num_layers,
        "num_classes": num_classes,
        "max_position_embeddings": max_position_embeddings,
        "dropout": dropout
    }
    logger.record_config(config)
    
    train_dataset = DogBarkTypeDataset(train_audio_path, train_label_path)
    eval_dataset = DogBarkTypeDataset(eval_audio_path, eval_label_path)
    unbalance_dataset = DogBarkTypeDataset(
        unbalance_audio_path, unbalance_label_path
    )
    
    train_dataset = ConcatDataset([train_dataset, unbalance_dataset])
    
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    eval_dataloader = DataLoader(eval_dataset, batch_size=32, shuffle=False)
    model = BERTForAudioClassification(feature_dim,
                                       d_model,
                                       num_heads,
                                       d_ff,
                                       num_layers,
                                       num_classes,
                                       max_position_embeddings,
                                       dropout)
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    # class_weights = calculate_class_weights(train_dataset)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=10, verbose=True)
    best_acc = 0.0
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        for features, labels in train_dataloader:
            features = features.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()

            outputs = model(features)
            loss = loss_fn(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        print(f"Epoch {epoch} loss: {epoch_loss}")
        logger.record_loss(epoch, epoch_loss)
        
        # evaluation
        if epoch % 20 == 19:
            model.eval()
            with torch.no_grad():
                all_preds = []
                all_labels = []
                for features, labels in eval_dataloader:
                    features = features.to(device)
                    labels = labels.to(device)
                    outputs = model(features)
                    preds = outputs > 0.5
                    all_preds.append(preds.cpu().numpy())
                    all_labels.append(labels.cpu().numpy())
                all_preds = np.concatenate(all_preds, axis=0)
                all_labels = np.concatenate(all_labels, axis=0)
                f1_micro = f1_score(all_labels, all_preds, average='micro')
                f1_macro = f1_score(all_labels, all_preds, average='macro')
                acc = accuracy_score(all_labels, all_preds)
                hamming_acc = hamming_loss(all_labels, all_preds)
                logger.record_eval(epoch, f1_micro, f1_macro, acc, hamming_acc)
                
                if acc > best_acc:
                    best_acc = acc
                    checkpoint_name = os.path.join(
                        logger.checkpoints_dir, f"model_{epoch}_{acc}.pth"
                    )
                    torch.save(model.state_dict(), checkpoint_name)
                
                print((f"Epoch {epoch} F1_Micro: {f1_micro}",
                      f"F1_Macro: {f1_macro}, Accuracy: {acc}",
                      f"Hamming Acc: {hamming_acc}"))
            scheduler.step(acc)
    logger.close()
       
       
def transcription_bark_type_main():
    
    train_label_path = "/data/dogs/sixbarktype_dataset/labels/train.csv"
    eval_label_path = "/data/dogs/sixbarktype_dataset/labels/eval.csv"
    unbalance_label_path = "/data/dogs/sixbarktype_dataset/labels/unbalanced.csv"
    
    num_epochs = 400
    vocab_size = 250  # Adjust based on the number of phonemes in your dataset
    d_model = 768  # Transformer model dimension
    num_heads = 12  # Number of attention heads
    d_ff = 768  # Feedforward hidden layer dimension
    num_layers = 6  # Number of transformer blocks
    num_classes = 6  # Number of classes for classification
    max_position_embeddings = 500  # Max number of phonemes in a sequence
    dropout = 0.1
    checkpoints_dir = f"barktype/trans_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    logger = Logger(checkpoints_dir)
    
    # Save the configuration
    config = {
        "num_epochs": num_epochs,
        "vocab_size": vocab_size,
        "d_model": d_model,
        "num_heads": num_heads,
        "d_ff": d_ff,
        "num_layers": num_layers,
        "num_classes": num_classes,
        "max_position_embeddings": max_position_embeddings,
        "dropout": dropout
    }
    logger.record_config(config)
    
    # Create datasets for phoneme sequences
    train_dataset = DogBarkTypeTranscriptionDataset(train_label_path)  
    eval_dataset = DogBarkTypeTranscriptionDataset(eval_label_path)
    unbalance_dataset = DogBarkTypeTranscriptionDataset(unbalance_label_path)
    
    train_dataset = ConcatDataset([train_dataset, unbalance_dataset])
    
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    eval_dataloader = DataLoader(eval_dataset, batch_size=32, shuffle=False)

    # Use the BERTForPhonemeClassification model
    model = BERTForPhonemeClassification(vocab_size,
                                         d_model,
                                         num_heads,
                                         d_ff,
                                         num_layers,
                                         num_classes,
                                         max_position_embeddings,
                                         dropout)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    # Define loss function and optimizer
    loss_fn = torch.nn.BCEWithLogitsLoss()  # For multi-label classification
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=10, verbose=True)
    
    best_acc = 0.0
    
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        
        for features, labels in train_dataloader:
            features = features.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass through the model
            outputs = model(features)
            loss = loss_fn(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        print(f"Epoch {epoch} loss: {epoch_loss}")
        logger.record_loss(epoch, epoch_loss)
        
        # Evaluate the model every 20 epochs
        if epoch % 20 == 19:
            model.eval()
            with torch.no_grad():
                all_preds = []
                all_labels = []
                
                for features, labels in eval_dataloader:
                    features = features.to(device)
                    labels = labels.to(device)
                    outputs = model(features)
                    preds = outputs > 0.5
                    all_preds.append(preds.cpu().numpy())
                    all_labels.append(labels.cpu().numpy())
                
                all_preds = np.concatenate(all_preds, axis=0)
                all_labels = np.concatenate(all_labels, axis=0)
                
                # Compute evaluation metrics
                f1_micro = f1_score(all_labels, all_preds, average='micro')
                f1_macro = f1_score(all_labels, all_preds, average='macro')
                acc = accuracy_score(all_labels, all_preds)
                hamming_acc = hamming_loss(all_labels, all_preds)
                
                # Log evaluation metrics
                logger.record_eval(epoch, f1_micro, f1_macro, acc, hamming_acc)
                
                # Save model if accuracy improves
                if acc > best_acc:
                    best_acc = acc
                    checkpoint_name = os.path.join(
                        logger.checkpoints_dir, f"model_{epoch}_{acc}.pth"
                    )
                    torch.save(model.state_dict(), checkpoint_name)
                
                print((f"Epoch {epoch} F1_Micro: {f1_micro}",
                      f"F1_Macro: {f1_macro}, Accuracy: {acc}",
                      f"Hamming Acc: {hamming_acc}"))
            
            # Step the learning rate scheduler
            scheduler.step(acc)
    
    logger.close()
        
        
def get_statistics_of_labels():
    train_label_path = "/data/dogs/sixbarktype_dataset/labels/train.csv"
    eval_label_path = "/data/dogs/sixbarktype_dataset/labels/eval.csv"
    unbalance_label_path = "/data/dogs/sixbarktype_dataset/labels/unbalanced.csv"
    
    with open(train_label_path, 'r') as f:
        train_data = pd.read_csv(f)
    m = 1


def get_x_name_y_():
    train_audio_path = "/data/dogs/sixbarktype_dataset/audios/train"
    eval_audio_path = "/data/dogs/sixbarktype_dataset/audios/eval"
    unbalance_audio_path = "/data/dogs/sixbarktype_dataset/audios/unbalanced"
    
    train_label_path = "/data/dogs/sixbarktype_dataset/labels/train.csv"
    eval_label_path = "/data/dogs/sixbarktype_dataset/labels/eval.csv"
    unbalance_label_path = "/data/dogs/sixbarktype_dataset/labels/unbalanced.csv"
    
    x, x_c, y = [], [], []
    for category in ["train", "eval", "unbalanced"]:
        audio_folder = f"/data/dogs/sixbarktype_dataset/audios/{category}"
        label_path = f"/data/dogs/sixbarktype_dataset/labels/{category}.csv"
        df = pd.read_csv(label_path)
        for _, row in df.iterrows():
            try:
                audio_name = row["filename"]
                labels = row["label"].split()
                
                audio_path = os.path.join(audio_folder, audio_name)
                if len(labels) > 1:
                    continue
                else:
                    label = labels[0]
                x.append(audio_name)
                x_c.append(category)
                y.append(label)
            except:
                print(f"Error processing {audio_path}")
                continue
    return x, x_c, y
                


def get_mfcc_feature():
    x, y = [], []
    for category in ["train", "eval", "unbalanced"]:
        audio_folder = f"/data/dogs/sixbarktype_dataset/audios/{category}"
        label_path = f"/data/dogs/sixbarktype_dataset/labels/{category}.csv"
        df = pd.read_csv(label_path)
        for _, row in df.iterrows():
            try:
                audio_name = row["filename"]
                labels = row["label"].split()
                if len(labels) > 1:
                    continue
                else:
                    label = labels[0]
                audio_path = os.path.join(audio_folder, audio_name)
                fea = get_mfcc_fea(audio_path).T.mean(axis=0)
                
                x.append(fea)
                y.append(label)
            except:
                print(f"Error processing {audio_path}")
                continue
    return x, y


def get_hubert_feature(hubert_version: str="hubert-250", cc: bool=False, mapping: str=None, phoneme_only: bool=False):
    folder = {
        "hubert-250": "/data/dogs/sixbarktype_dataset/transcriptions/1012-2024130k_11th_km250",
        "hubert-500": "/data/dogs/sixbarktype_dataset/transcriptions/1013-2024130k_11th_km500"
    }
    cluster_center = {
        "hubert-250": "/data/xingyuan/animal_language/v2/out/fairseq/six_balance/iter3/130k_11th_km250.pkl",
        "hubert-500": "/data/xingyuan/animal_language/v2/out/fairseq/six_balance/iter3/130k_11th_km500.pkl"
    }
    mappings = {
        "hubert-250-t5-p15": "/home/sinong/ALP/AniLang/PhonemeFinding/checkpoints/phoneme_refined/241012_224401/mapping_phoneme.json",
        "hubert-250-t5-p10": "/home/sinong/ALP/AniLang/PhonemeFinding/checkpoints/phoneme_refined/241012_234556/mapping_phoneme.json",
        "hubert-250-t5-p5": "/home/sinong/ALP/AniLang/PhonemeFinding/checkpoints/phoneme_refined/241013_001019/mapping_phoneme.json",
        "hubert-500-t5-p15": "/home/sinong/ALP/AniLang/PhonemeFinding/checkpoints/phoneme_refined/241013_180004/mapping_phoneme.json",
        "hubert-500-t5-p10": "/home/sinong/ALP/AniLang/PhonemeFinding/checkpoints/phoneme_refined/241013_174132/mapping_phoneme.json",
        "hubert-500-t5-p5": "/home/sinong/ALP/AniLang/PhonemeFinding/checkpoints/phoneme_refined/241013_173631/mapping_phoneme.json",
    }
    configs = {
        "hubert-250-t5-p15": "/home/sinong/ALP/AniLang/PhonemeFinding/checkpoints/phoneme_refined/241012_224401/config.json",
        "hubert-250-t5-p10": "/home/sinong/ALP/AniLang/PhonemeFinding/checkpoints/phoneme_refined/241012_234556/config.json",
        "hubert-250-t5-p5": "/home/sinong/ALP/AniLang/PhonemeFinding/checkpoints/phoneme_refined/241013_001019/config.json",
        "hubert-500-t5-p15": "/home/sinong/ALP/AniLang/PhonemeFinding/checkpoints/phoneme_refined/241013_180004/config.json",
        "hubert-500-t5-p10": "/home/sinong/ALP/AniLang/PhonemeFinding/checkpoints/phoneme_refined/241013_174132/config.json",
        "hubert-500-t5-p5": "/home/sinong/ALP/AniLang/PhonemeFinding/checkpoints/phoneme_refined/241013_173631/config.json",
    }
    if cc:
        cluster_center = load(cluster_center[hubert_version]).cluster_centers_
    if mapping:
        with open(configs[mapping], "r") as f:
            config = json.load(f)
        found_phoneme = config["found_phonemes"]
        with open(mappings[mapping], "r") as f:
            mapping_dict = json.load(f)
    
    x_fea, x_c, yy = get_x_name_y_()
    x, y = [], []
    for filename, category, label in zip(x_fea, x_c, yy):
        try:
            if cc:
                # cluster_center = load(cluster_center[hubert_version]).cluster_centers_
                trans_path = f"{folder[hubert_version]}/{category}/label_{filename.split('.')[0]}.npy"
                trans = np.load(trans_path)
                trans = [int(t) for t in trans]
                new_trans = []
                if mapping:
                    for idx, t in enumerate(trans):
                        if str(t) in mapping_dict:
                            new_trans.append(int(mapping_dict[str(t)]))
                        else:
                            new_trans.append(t)
                    trans = new_trans
                    if phoneme_only:
                        feat = np.zeros((768))
                        n = 0
                        for t in trans:
                            if str(t) in found_phoneme:
                                feat += cluster_center[t]
                                n += 1
                        if n == 0:
                            feat = np.zeros((768))
                        else:
                            feat /= n
                    else:
                        feat = np.zeros((len(trans), 768))
                        for idx, t in enumerate(trans):
                            feat[idx] = cluster_center[t]
                        feat = feat.mean(axis=0)
                else:
                    feat = np.zeros((len(trans), 768))
                    for idx, t in enumerate(trans):
                        feat[idx] = cluster_center[t]
                    feat = feat.mean(axis=0)
            else:
                feat_path = f"{folder[hubert_version]}/{category}/feat_{filename.split('.')[0]}.npy"
                feat = np.load(feat_path)
                feat = feat.mean(axis=0)
            x.append(feat)
            y.append(label)
        except Exception as e:
            print(f"Error processing:{e}, {filename}")
            continue
    assert len(x) == len(y)
    return x, y


def hubert_feature_ml_method(hubert_version: str="hubert-250", cc: bool=False, mapping: str=None, phoneme_only: bool=False, mfcc: bool=False): 
    
    save_folder = "/home/sinong/ALP/AniLang/Evaluation/checkpoints/barktype/no_scaling"
    os.makedirs(save_folder, exist_ok=True)   
    # Step 1: Get features and labels
    if mfcc:
        x, y = get_mfcc_feature()
    else:
        x, y = get_hubert_feature(hubert_version, cc, mapping, phoneme_only)

    # Step 2: Convert multi-label target to binary format

    mlb = MultiLabelBinarizer()
    y_bin = mlb.fit_transform(y)


    # Step 3: Train-test split
    X_train, X_test, y_train, y_test = train_test_split(x, y_bin, test_size=0.2, random_state=42)

    # Step 4: Standardize features (useful for Logistic Regression)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Step 5: Initialize models
    models = {
        "Logistic Regression": OneVsRestClassifier(LogisticRegression(solver='liblinear')),
        "Random Forest": OneVsRestClassifier(RandomForestClassifier(n_estimators=100, random_state=42)),
        "XGBoost": OneVsRestClassifier(XGBClassifier(eval_metric='mlogloss', use_label_encoder=False))
    }

    # Step 6: Train, evaluate, and store results
    results = []

    def train_and_evaluate(model, model_name):
        print(f"\nTraining {model_name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)
        print(f"{model_name} Accuracy: {accuracy:.2f}")

        # Generate classification report
        report = classification_report(y_test, y_pred, target_names=mlb.classes_, output_dict=True)
        print(f"Classification Report for {model_name}:\n")
        print(classification_report(y_test, y_pred, target_names=mlb.classes_))

        # Store results in a structured way for CSV
        for label, metrics in report.items():
            if isinstance(metrics, dict):  # Skip the 'accuracy' field in the report
                results.append({
                    "Model": model_name,
                    "Label": label,
                    "Precision": metrics.get("precision", 0.0),
                    "Recall": metrics.get("recall", 0.0),
                    "F1-Score": metrics.get("f1-score", 0.0),
                    "Support": metrics.get("support", 0),
                    "Overall Accuracy": accuracy
                })

    # Train and evaluate each model
    for model_name, model in models.items():
        train_and_evaluate(model, model_name)

    # Step 7: Save results to a CSV file
    df = pd.DataFrame(results)
    # Save results to CSV
    if mfcc:
        file_name=f"mfcc_classification_results.csv"
    else:
        file_name = f"{hubert_version}{'_cc' if cc else ''}_{mapping if mapping else 'no_mapping'}{'_only' if phoneme_only else ''}_classification_results.csv"
    df.to_csv(os.path.join(save_folder, file_name), index=False)
    print("\nResults have been saved to classification_results.csv")
    

def hubert_feature_ml_method_single(
    hubert_version: str = "hubert-250", 
    cc: bool = False, 
    mapping: str = None, 
    phoneme_only: bool = False, 
    mfcc: bool = False
):
    # Set up the save folder
    save_folder = "/home/sinong/ALP/AniLang/Evaluation/checkpoints/barktype/single"
    os.makedirs(save_folder, exist_ok=True)

    # Step 1: Get features and labels
    if mfcc:
        x, y = get_mfcc_feature()
    else:
        x, y = get_hubert_feature(hubert_version, cc, mapping, phoneme_only)

    # Step 2: Encode string labels to integers
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)  # Encode labels

    # Step 3: Train-test split
    X_train, X_test, y_train, y_test = train_test_split(x, y_encoded, test_size=0.2, random_state=42)

    # Step 4: Standardize features (useful for Logistic Regression)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Step 5: Initialize models
    models = {
        "Logistic Regression": LogisticRegression(solver='liblinear'),
        "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=20, random_state=42),
        "XGBoost": XGBClassifier(n_estimators=300, eval_metric='mlogloss', use_label_encoder=False)
    }

    # Step 6: Train, evaluate, and store results
    results = []

    def train_and_evaluate(model, model_name):
        print(f"\nTraining {model_name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)
        print(f"{model_name} Accuracy: {accuracy:.2f}")

        # Generate classification report
        report = classification_report(
            y_test, y_pred, target_names=label_encoder.classes_, output_dict=True
        )
        print(f"Classification Report for {model_name}:\n")
        print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

        # Store results in a structured way for CSV
        for label, metrics in report.items():
            if isinstance(metrics, dict):  # Skip fields like 'accuracy'
                results.append({
                    "Model": model_name,
                    "Label": label,
                    "Precision": metrics.get("precision", 0.0),
                    "Recall": metrics.get("recall", 0.0),
                    "F1-Score": metrics.get("f1-score", 0.0),
                    "Support": metrics.get("support", 0),
                    "Overall Accuracy": accuracy
                })

    # Train and evaluate each model
    for model_name, model in models.items():
        train_and_evaluate(model, model_name)

    # Step 7: Save results to a CSV file
    df = pd.DataFrame(results)
    if mfcc:
        file_name = "mfcc_classification_results.csv"
    else:
        file_name = (
            f"{hubert_version}{'_cc' if cc else ''}_{mapping if mapping else 'no_mapping'}"
            f"{'_only' if phoneme_only else ''}_classification_results.csv"
        )
    df.to_csv(os.path.join(save_folder, file_name), index=False)
    print(f"\nResults have been saved to {file_name}")     


def acoustic_feature_ml_method(fea_mode: str):
    train_audio_path = "/data/dogs/sixbarktype_dataset/audios/train"
    eval_audio_path = "/data/dogs/sixbarktype_dataset/audios/eval"
    unbalance_audio_path = "/data/dogs/sixbarktype_dataset/audios/unbalanced"
    
    train_label_path = "/data/dogs/sixbarktype_dataset/labels/train.csv"
    eval_label_path = "/data/dogs/sixbarktype_dataset/labels/eval.csv"
    unbalance_label_path = "/data/dogs/sixbarktype_dataset/labels/unbalanced.csv"
    
    x, y = [], []
    for category in ["train", "eval", "unbalanced"]:
        audio_folder = f"/data/dogs/sixbarktype_dataset/audios/{category}"
        label_path = f"/data/dogs/sixbarktype_dataset/labels/{category}.csv"
        df = pd.read_csv(label_path)
        for _, row in df.iterrows():
            try:
                audio_name = row["filename"]
                labels = row["label"].split()
                audio_path = os.path.join(audio_folder, audio_name)
                if fea_mode == "mfcc":
                    fea = get_mfcc_fea(audio_path).T.mean(axis=0)
                elif fea_mode == "ori-hubert-250":
                    feature_folder = f"/data/dogs/sixbarktype_dataset/transcriptions/1012-2024130k_11th_km250/{category}"
                    fea = np.load(os.path.join(feature_folder, f"feat_{audio_name.split('.')[0]}.npy"))
                    fea = fea.mean(axis=0)
                elif fea_mode == "ori-hubert-250-p15":
                    trans_folder = f"/data/dogs/sixbarktype_dataset/transcriptions/1012-2024130k_11th_km250/{category}"
                    kmmodel = "/data/xingyuan/animal_language/v2/out/fairseq/six_balance/iter3/130k_11th_km250.pkl"
                    mapping_file_path = "/home/sinong/ALP/AniLang/PhonemeFinding/checkpoints/phoneme_refined/241012_224401/mapping_phoneme.json"
                    with open(mapping_file_path, 'r') as f:
                        mapping = json.load(f)
                    cluster_centers = load(kmmodel).cluster_centers_
                    trans = np.load(os.path.join(trans_folder, f"label_{audio_name.split('.')[0]}.npy"))
                    trans = [int(tran) for tran in trans]
                    # build the feature
                    fea = np.zeros((len(trans), 768))
                    for i, tran in enumerate(trans):
                        if tran in mapping:
                            replaced_tran = mapping[tran]
                        else:
                            replaced_tran = tran
                        fea[i] = cluster_centers[replaced_tran]
                    fea = fea.mean(axis=0)
                elif fea_mode == "ori-hubert-250-p10":
                    trans_folder = f"/data/dogs/sixbarktype_dataset/transcriptions/1012-2024130k_11th_km250/{category}"
                    kmmodel = "/data/xingyuan/animal_language/v2/out/fairseq/six_balance/iter3/130k_11th_km250.pkl"
                    mapping_file_path = "/home/sinong/ALP/AniLang/PhonemeFinding/checkpoints/phoneme_refined/241013_103415/mapping_phoneme.json"
                    with open(mapping_file_path, 'r') as f:
                        mapping = json.load(f)
                    cluster_centers = load(kmmodel).cluster_centers_
                    trans = np.load(os.path.join(trans_folder, f"label_{audio_name.split('.')[0]}.npy"))
                    trans = [int(tran) for tran in trans]
                    # build the feature
                    fea = np.zeros((len(trans), 768))
                    for i, tran in enumerate(trans):
                        if tran in mapping:
                            replaced_tran = mapping[tran]
                        else:
                            replaced_tran = tran
                        fea[i] = cluster_centers[replaced_tran]
                    fea = fea.mean(axis=0)
                
                elif fea_mode == "cluster-center":
                    trans_folder = f"/data/dogs/sixbarktype_dataset/transcriptions/1012-2024130k_11th_km250/{category}"
                    kmmodel = "/data/xingyuan/animal_language/v2/out/fairseq/six_balance/iter3/130k_11th_km250.pkl"

                    cluster_centers = load(kmmodel).cluster_centers_
                    trans = np.load(os.path.join(trans_folder, f"label_{audio_name.split('.')[0]}.npy"))
                    trans = [int(tran) for tran in trans]
                    # build the feature
                    fea = np.zeros((len(trans), 768))
                    for i, tran in enumerate(trans):
                        replaced_tran = tran
                        fea[i] = cluster_centers[replaced_tran]
                    fea = fea.mean(axis=0)
                
                elif fea_mode == "only-phoneme":
                    trans_folder = f"/data/dogs/sixbarktype_dataset/transcriptions/1012-2024130k_11th_km250/{category}"
                    kmmodel = "/data/xingyuan/animal_language/v2/out/fairseq/six_balance/iter3/130k_11th_km250.pkl"
                    mapping_file_path = "/home/sinong/ALP/AniLang/PhonemeFinding/checkpoints/phoneme_refined/241012_224401/mapping_phoneme.json"
                    config_file_path = "/home/sinong/ALP/AniLang/PhonemeFinding/checkpoints/phoneme_refined/241012_224401/config.json"
                    with open(config_file_path, 'r') as f:
                        config = json.load(f)
                    with open(mapping_file_path, 'r') as f:
                        mapping = json.load(f)
                    found_phoneme = config["found_phonemes"]
                    cluster_centers = load(kmmodel).cluster_centers_
                    trans = np.load(os.path.join(trans_folder, f"label_{audio_name.split('.')[0]}.npy"))
                    trans = [int(tran) for tran in trans]
                    # build the feature
                    fea = np.zeros((768))
                    num = 0
                    for i, tran in enumerate(trans):
                        if tran in mapping:
                            replaced_tran = mapping[tran]
                        else:
                            replaced_tran = tran
                        if str(replaced_tran) in found_phoneme:
                            fea += cluster_centers[replaced_tran]
                            num += 1
                    if num == 0:
                        fea = np.zeros((768))
                    else:
                        fea = fea / num
                
                x.append(fea)
                y.append(labels)
            except:
                print(f"Error processing {audio_path}")
                continue
        
    # Step 1: Convert multi-label target to binary format
    mlb = MultiLabelBinarizer()
    y_bin = mlb.fit_transform(y)

    # Step 2: Train-test split
    X_train_mfcc, X_test_mfcc, y_train_mfcc, y_test_mfcc = train_test_split(x, y_bin, test_size=0.2, random_state=42)

    # Step 3: Standardize features (useful for LR)
    # scaler = StandardScaler()
    # X_train_mfcc = scaler.fit_transform(X_train_mfcc)
    # X_test_mfcc = scaler.transform(X_test_mfcc)

    # Step 4: Initialize models
    lr = OneVsRestClassifier(LogisticRegression(solver='liblinear'))
    rf = OneVsRestClassifier(RandomForestClassifier(n_estimators=100, random_state=42))
    xgb = OneVsRestClassifier(XGBClassifier(eval_metric='mlogloss', use_label_encoder=False))
    
    # Step 5: Train and evaluate models
    def train_and_evaluate(model, model_name):
        print(f"\nTraining {model_name}...")
        model.fit(X_train_mfcc, y_train_mfcc)
        y_pred = model.predict(X_test_mfcc)
        print(f"Classification Report for {model_name}:\n")
        print(classification_report(y_test_mfcc, y_pred, target_names=mlb.classes_))
        
    train_and_evaluate(rf, "Random Forest")
    train_and_evaluate(xgb, "XGBoost")
    train_and_evaluate(lr, "Logistic Regression")

   
def main():
    
    # bark_type_main()
    # get_statistics_of_labels()
    # transcription_bark_type_main()
    # acoustic_feature_ml_method("ori-hubert-250-p10")
    # acoustic_feature_ml_method("cluster-center")
    # acoustic_feature_ml_method("only-phoneme")
    # hubert_feature_ml_method(mfcc=True)
    # hubert_feature_ml_method("hubert-250", cc=False, mapping=None, phoneme_only=False)
    # hubert_feature_ml_method("hubert-250", cc=True, mapping=None, phoneme_only=False)
    # hubert_feature_ml_method("hubert-250", cc=True, mapping="hubert-250-t5-p15", phoneme_only=True)
    # hubert_feature_ml_method("hubert-250", cc=True, mapping="hubert-250-t5-p10", phoneme_only=True)
    # hubert_feature_ml_method("hubert-250", cc=True, mapping="hubert-250-t5-p5", phoneme_only=True)
    # hubert_feature_ml_method("hubert-250", cc=True, mapping="hubert-250-t5-p5", phoneme_only=False)
    
    # hubert_feature_ml_method("hubert-500", cc=True, mapping="hubert-500-t5-p15", phoneme_only=True)
    # hubert_feature_ml_method("hubert-500", cc=True, mapping="hubert-500-t5-p10", phoneme_only=True)
    # hubert_feature_ml_method("hubert-500", cc=True, mapping="hubert-500-t5-p5", phoneme_only=True)
    # hubert_feature_ml_method("hubert-500", cc=True, mapping="hubert-500-t5-p5", phoneme_only=False)
    # hubert_feature_ml_method("hubert-500", cc=False, mapping=None, phoneme_only=False)
    # hubert_feature_ml_method("hubert-500", cc=True, mapping=None, phoneme_only=False)
    
    hubert_feature_ml_method_single(mfcc=True)
    hubert_feature_ml_method_single("hubert-250", cc=False, mapping=None, phoneme_only=False)
    hubert_feature_ml_method_single("hubert-250", cc=True, mapping=None, phoneme_only=False)
    hubert_feature_ml_method_single("hubert-250", cc=True, mapping="hubert-250-t5-p15", phoneme_only=True)
    hubert_feature_ml_method_single("hubert-250", cc=True, mapping="hubert-250-t5-p10", phoneme_only=True)
    hubert_feature_ml_method_single("hubert-250", cc=True, mapping="hubert-250-t5-p5", phoneme_only=True)
    hubert_feature_ml_method_single("hubert-250", cc=True, mapping="hubert-250-t5-p5", phoneme_only=False) 


if __name__ == '__main__':
    main()
