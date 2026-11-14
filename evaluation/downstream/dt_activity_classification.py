import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
import numpy as np
import pandas as pd
from joblib import load
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from constant import get_mfcc_fea


def get_dataset():
    meta_file_path = "/data/dogs/sixbarktype_dataset/labels/video_labels.csv"
    metadata = pd.read_csv(meta_file_path)
    metadata = metadata[metadata["have_labels"] == True]
    metadata = metadata[metadata["is_invalid"] == False]
    
    x_filename = []
    x_category = []
    y_cause = []
    y_posture = []
    y_ear = []
    y_tail = []
    
    for idx, row in metadata.iterrows():
        filename = row["video_filename"]
        category = row["category"]
        cause = row["cause"]
        posture = row["body_posture"]
        ear = row["ear"]
        tail = row["tail"]
        x_filename.append(filename)
        x_category.append(category)
        y_cause.append(cause)
        y_posture.append(posture)
        y_ear.append(ear)
        y_tail.append(tail)
    
    return {
        "filename": x_filename,
        "category": x_category,
        "cause": y_cause,
        "posture": y_posture,
        "ear": y_ear,
        "tail": y_tail
    }


def get_mfcc_feat():
    data = get_dataset()
    x_filename = data["filename"]
    x_category = data["category"]
    x = []
    for idx, (filename, category) in enumerate(zip(x_filename, x_category)):
        audio_path = f"/data/dogs/sixbarktype_dataset/audios/{category}/{filename.split('.')[0]}.wav"
        feat = get_mfcc_fea(audio_path).T.mean(axis=0)
        x.append(feat)
    x_cause, y_cause = [], []
    x_posture, y_posture = [], []
    x_ear, y_ear = [], []
    x_tail, y_tail = [], []
    for _, (x_feat, cause, posture, ear, tail) in enumerate(zip(x, data["cause"], data["posture"], data["ear"], data["tail"])):
        if isinstance(cause, str):
            x_cause.append(x_feat)
            y_cause.append(cause)
        if isinstance(posture, str):
            x_posture.append(x_feat)
            y_posture.append(posture)
        if isinstance(ear, str):
            x_ear.append(x_feat)
            y_ear.append(ear)
        if isinstance(tail, str):
            x_tail.append(x_feat)
            y_tail.append(tail)
    return {
        "cause": (x_cause, y_cause),
        "posture": (x_posture, y_posture),
        "ear": (x_ear, y_ear),
        "tail": (x_tail, y_tail)
    }


def get_hubert_feat(hubert_version: str="hubert-250", cc: bool=False, mapping: str=None, phoneme_only: bool=False):
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
        
    data = get_dataset()
    x_filename = data["filename"]
    x_category = data["category"]
    cause = data["cause"]
    posture = data["posture"]
    ear = data["ear"]
    tail = data["tail"]
    x, new_cause, new_posture, new_ear, new_tail = [], [], [], [], []
    for idxidx, (filename, category) in enumerate(zip(x_filename, x_category)):
        try:
            if cc:
                # cluster_center = load(cluster_center[hubert_version]).cluster_centers_
                trans_path = f"{folder[hubert_version]}/{category}/label_{filename.split('.')[0]}.npy"
                trans = np.load(trans_path)
                trans = [int(t) for t in trans]
                new_trans = []
                if mapping:
                    with open(mappings[mapping], "r") as f:
                        mapping_dict = json.load(f)
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
            new_cause.append(cause[idxidx])
            new_posture.append(posture[idxidx])
            new_ear.append(ear[idxidx])
            new_tail.append(tail[idxidx])
        except Exception as e:
            print(f"Error processing:{e}, {filename}")
            continue
    x_cause, y_cause = [], []
    x_posture, y_posture = [], []
    x_ear, y_ear = [], []
    x_tail, y_tail = [], []
    
    for _, (x_feat, cause, posture, ear, tail) in enumerate(zip(x, new_cause, new_posture, new_ear, new_tail)):
        if isinstance(cause, str):
            x_cause.append(x_feat)
            y_cause.append(cause)
        if isinstance(posture, str):
            x_posture.append(x_feat)
            y_posture.append(posture)
        if isinstance(ear, str):
            x_ear.append(x_feat)
            y_ear.append(ear)
        if isinstance(tail, str):
            x_tail.append(x_feat)
            y_tail.append(tail)
    return {
        "cause": (x_cause, y_cause),
        "posture": (x_posture, y_posture),
        "ear": (x_ear, y_ear),
        "tail": (x_tail, y_tail)
    }
    


def activity_classification_ml(hubert_version: str="hubert-250", cc: bool=False, mapping: str=None, phoneme_only: bool=False, mfcc: bool=False):
    save_folder = "/home/sinong/ALP/AniLang/Evaluation/checkpoints/activity"
    # sub_folder_name = datetime.now().strftime("%Y%m%d-%H%M%S")
    # save_folder = os.path.join(save_folder, sub_folder_name)
    os.makedirs(save_folder, exist_ok=True)
    
    # Get feature data for the different categories
    if mfcc:
        data = get_mfcc_feat()
    else:
        data = get_hubert_feat(hubert_version=hubert_version, cc=cc, mapping=mapping, phoneme_only=phoneme_only)

    # Define models for each category
    rf_models = {key: RandomForestClassifier(random_state=42) for key in data.keys()}
    xgb_models = {key: XGBClassifier(eval_metric="mlogloss", use_label_encoder=False) for key in data.keys()}
    lr_models = {key: LogisticRegression(solver="liblinear") for key in data.keys()}

    # Store results
    results = []
    label_encoders = {}  # Store label encoders for each category to decode later

    # Train and evaluate models for each task (cause, posture, ear, tail)
    for key, (x, y) in data.items():
        print(f"\nTraining models for {key}...")

        # Convert features and labels to numpy arrays
        x = np.array(x)

        # Encode string labels into integers
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)  # Encode labels to integers
        label_encoders[key] = le  # Store the encoder for decoding predictions

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(x, y_encoded, test_size=0.2, random_state=42)

        # Train the models
        rf_models[key].fit(X_train, y_train)
        xgb_models[key].fit(X_train, y_train)
        lr_models[key].fit(X_train, y_train)

        # Predict on test data
        rf_pred = rf_models[key].predict(X_test)
        xgb_pred = xgb_models[key].predict(X_test)
        lr_pred = lr_models[key].predict(X_test)

        # Decode predictions back to original string labels
        rf_pred_labels = le.inverse_transform(rf_pred)
        xgb_pred_labels = le.inverse_transform(xgb_pred)
        lr_pred_labels = le.inverse_transform(lr_pred)
        y_test_labels = le.inverse_transform(y_test)

        # Evaluate the models
        rf_acc = accuracy_score(y_test_labels, rf_pred_labels)
        xgb_acc = accuracy_score(y_test_labels, xgb_pred_labels)
        lr_acc = accuracy_score(y_test_labels, lr_pred_labels)

        print(f"\n{key} - RandomForest Accuracy: {rf_acc:.2f}")
        print(f"{key} - XGBoost Accuracy: {xgb_acc:.2f}")
        print(f"{key} - LogisticRegression Accuracy: {lr_acc:.2f}")

        # Store classification reports in a structured way
        results.append({
            "Category": key,
            "Model": "RandomForest",
            "Accuracy": rf_acc,
            "Report": classification_report(y_test_labels, rf_pred_labels, zero_division=0, output_dict=True)
        })
        results.append({
            "Category": key,
            "Model": "XGBoost",
            "Accuracy": xgb_acc,
            "Report": classification_report(y_test_labels, xgb_pred_labels, zero_division=0, output_dict=True)
        })
        results.append({
            "Category": key,
            "Model": "LogisticRegression",
            "Accuracy": lr_acc,
            "Report": classification_report(y_test_labels, lr_pred_labels, zero_division=0, output_dict=True)
        })

    # Convert results to a DataFrame for saving to CSV
    formatted_results = []
    for result in results:
        report = result["Report"]
        for label, metrics in report.items():
            if isinstance(metrics, dict):
                formatted_results.append({
                    "Category": result["Category"],
                    "Model": result["Model"],
                    "Accuracy": result["Accuracy"],
                    "Label": label,
                    **metrics
                })

    df = pd.DataFrame(formatted_results)
    
    # Save results to CSV
    if mfcc:
        file_name = "mfcc_classification_results.csv"
    else:
        file_name = f"{hubert_version}{'_cc' if cc else ''}_{mapping if mapping else 'no_mapping'}{'_only' if phoneme_only else ''}_classification_results.csv"
    df.to_csv(os.path.join(save_folder, file_name), index=False)
    print("\nResults have been saved to classification_results.csv")


if __name__ == "__main__":
    activity_classification_ml(mfcc=True)
    
    activity_classification_ml(hubert_version="hubert-250", cc=False, mapping=None, phoneme_only=False)
    activity_classification_ml(hubert_version="hubert-250", cc=True, mapping=None, phoneme_only=False)
    activity_classification_ml(hubert_version="hubert-250", cc=True, mapping="hubert-250-t5-p5", phoneme_only=False)
    activity_classification_ml(hubert_version="hubert-250", cc=True, mapping="hubert-250-t5-p5", phoneme_only=True)
    activity_classification_ml(hubert_version="hubert-250", cc=True, mapping="hubert-250-t5-p10", phoneme_only=True)
    activity_classification_ml(hubert_version="hubert-250", cc=True, mapping="hubert-250-t5-p15", phoneme_only=True)
    
    activity_classification_ml(hubert_version="hubert-500", cc=False, mapping=None, phoneme_only=False)
    activity_classification_ml(hubert_version="hubert-500", cc=True, mapping=None, phoneme_only=False)
    activity_classification_ml(hubert_version="hubert-500", cc=True, mapping="hubert-500-t5-p5", phoneme_only=False)
    activity_classification_ml(hubert_version="hubert-500", cc=True, mapping="hubert-500-t5-p15", phoneme_only=True)
    activity_classification_ml(hubert_version="hubert-500", cc=True, mapping="hubert-500-t5-p10", phoneme_only=True)
    activity_classification_ml(hubert_version="hubert-500", cc=True, mapping="hubert-500-t5-p5", phoneme_only=True)

