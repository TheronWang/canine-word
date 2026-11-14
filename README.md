# Canine Phonetic Alphabet

Code for the paper **Toward Automatic Discovery of a Canine Phonetic Alphabet** (ACL 2025, Outstanding Paper).
Theron S. Wang, Xingyuan Li, Hridayesh Lekhak, Tuan M. Dang, Mengyue Wu, Kenny Q. Zhu.
[Paper](https://aclanthology.org/2025.acl-long.451/)

From 6,235 hours of YouTube dog videos, the pipeline cuts 306,233 dog vocal segments (152 h). It transcribes them into 250 initial vocal units with a dog-trained HuBERT. It then refines those units with minimal pairs into **105 cognitive vocal units** (the canine phonetic alphabet) and **69 vocal patterns**.

```
videos ─► 1 preprocessing: PANNs first cut ─► AudioSep ─► BEATs SED (pause > 0.5 s)
       ─► 2 transcription: 2-stage Dog-HuBERT ─► layer-11 k-means (K = 250)
       ─► 3 refinement: energy segmentation (ρ = 3) ─► minimal pairs ─► JS-divergence merging
       ─► cognitive vocal units + vocal patterns
```

## Models and data

Large files are on the Hugging Face Hub:

- [`ArlingtonCL2/canine-phonetic-models`](https://huggingface.co/ArlingtonCL2/canine-phonetic-models): the BEATs dog detector, Dog-HuBERT (`dog_hubert_stage2_130k.pt`), the K = 250 k-means, and the stage-1/2 HuBERT target k-means.
- [`ArlingtonCL2/canine-phonetic-data`](https://huggingface.co/datasets/ArlingtonCL2/canine-phonetic-data) (dataset): transcriptions of all 306,233 vocal segments, with timestamps and candidate vocal patterns, plus the detector's segment boundaries and video durations.

```bash
pip install huggingface_hub
python download.py            # writes ./checkpoints and ./data
```

Audio and video are not redistributed. Segment names give the source YouTube channel and video.

## Code

The scripts are the ones used for the paper, kept as they were run. Many contain absolute paths from our servers, both in their defaults and in the example commands in their docstrings. Change these to your own paths.

| Folder | Paper section | What it does |
|---|---|---|
| [`1_preprocessing/first_cut/`](1_preprocessing/first_cut) | §2.1 | `first_cut.py`: pretrained [PANNs](https://github.com/qiuqiangkong/audioset_tagging_cnn) Cnn14_DecisionLevelMax (Dog class, threshold 0.05, ±1 s) finds regions with dog sounds. |
| [`1_preprocessing/audiosep/`](1_preprocessing/audiosep) | §2.1 | `separate.py` runs [AudioSep](https://github.com/Audio-AGI/AudioSep) with the text query `"Dog"` on each region. |
| [`1_preprocessing/sed/train/`](1_preprocessing/sed/train) | §2.1, §3 | Fine-tunes the DCASE 2023 Task 4 baseline with a pretrained BEATs encoder (`beats_train_stage1.py`, `beats_stage1_trainer.py`, `CRNN_beats*.py`). `frozen_udesed.yaml` holds the settings: a 4-layer RNN, 256-dim CNN output, lr 0.001, 200 epochs, batch size 4. The same file configures inference. |
| [`1_preprocessing/sed/inference/`](1_preprocessing/sed/inference) | §2.1 | `second_cut.py` runs the detector (checkpoint epoch 109 / step 6160, threshold 0.6713) and merges detections less than 0.5 s apart into vocal segments. `test.py` evaluates the checkpoint. |
| [`1_preprocessing/dataset/`](1_preprocessing/dataset) | Table 2 | `data_statistics.py::get_statistics_data` writes per-video durations. `segment_statistics.py` counts segments and hours per breed. |
| [`2_transcription/`](2_transcription) | §2.2, §3 | `balance_data.py` cuts clips longer than 5 s into 5 s windows with a 1 s overlap; clips shorter than 0.35 s are dropped. `hubert_config/` holds the HuBERT training configs. `kmeans/` has the cluster sweep (`find_cluster_iter3.py`, its output `find_cluster_from_iter2_130k_11th.txt`) and the elbow plot (`plot_elbow.py`). `transcribe.py` labels every segment with layer-11 features and K = 250. |
| [`3_refinement/`](3_refinement) | §2.3, Alg. 1–2 | `word_finder_energy.py`: energy-based segmentation into candidate vocal patterns with [auditok](https://github.com/amsehili/auditok) (threshold = ρ × RMS). `phoneme_finder.py`: the `PhonemeFinder` class. It finds minimal pairs, computes contextual JS divergence, merges units below the percentile threshold (union-find) and filters units and patterns until nothing changes. |
| [`evaluation/abx/`](evaluation/abx) | §4.1.2, Table 4 | `phone_eval.py` generates ABX triplets and scores answers. `samples/` holds the answer keys of the 500 initial-unit and 210 final-unit triplets. |
| [`evaluation/downstream/`](evaluation/downstream) | §4.1.3, Tables 6–7 | `inference.py` transcribes the AudioSet bark clips. `dt_aduio_classification.py` does bark-type classification and `dt_activity_classification.py` does surroundings/ear/tail classification, each with LR, RF and XGBoost on MFCC, frame, vocal-unit and cognitive-unit features. `results/context/` holds the Table 7 output. |
| [`evaluation/vocal_patterns/`](evaluation/vocal_patterns) | §4.2, Figs 5–6, Table 11 | `word_meaning.py` takes 3 frames from 3 s before to 3 s after each pattern occurrence (`process_sentence`). It maps the Janus-Pro descriptions to categories with the CSVs in `category_mapping/` (`merge_word_meaning`) and computes the relative frequency of Eq. 1 (`wordMeaningAnalyzer.generate_word_meaning_distribution_plot`). `plot_word_distribution2` draws Figure 6. `janus_describe_frames.py` runs Janus-Pro-7B on the frames. `word_analysis.ipynb` computes the pattern durations in Table 11. |
| [`evaluation/dataset_table2.py`](evaluation/dataset_table2.py) | Table 2 | Recomputes Table 2 from the released data. |
| [`results/`](results) | Table 11 | `vocal_patterns_69_durations.json`: mean and std duration of the 69 vocal patterns. |

## Step by step

### 1. Preprocessing

1. **First cut.** Put `first_cut/` inside a clone of audioset_tagging_cnn as `work/` and run `first_cut.py` on the raw video audio.
2. **AudioSep.** Put `audiosep/` inside a clone of AudioSep as `work/` and run `separate.py` with `--text Dog`.
3. **Sound event detection.** The detector code builds on [ATST-SED](https://github.com/Audio-WestlakeU/ATST-SED). Place `sed/train/*` under its `train/`, the `CRNN_beats*.py` files under `desed_task/nnet/`, and `sed/inference/*` in a work folder. The pretrained BEATs encoder is `BEATs_iter3_plus_AS2M.pt`. The detector is post-trained on more than 9,000 s of manually labelled pure dog vocalizations with a learning rate of 0.001 for 200 epochs, batch size 4, on two NVIDIA RTX 4090 GPUs (about 6 hours). The released checkpoint `beats_sed_dog.ckpt` reaches an F1 score of 0.8556 on the test set.

### 2. Transcription (Dog-HuBERT)

We trained a three-stage HuBERT with the standard [fairseq HuBERT recipe](https://github.com/facebookresearch/fairseq/tree/main/examples/hubert) and the configs in `2_transcription/hubert_config/`:

| Stage | Clusters | Released file |
|---|---|---|
| 1 | 100 (MFCC k-means) | `stage1_targets_mfcc_k100.pkl` |
| 2 | 200 | `stage2_targets_layer6_k200.pkl`, model `dog_hubert_stage2_130k.pt` |
| 3 | 250, chosen with the elbow method (`kmeans/`) | `kmeans_layer11_k250.pkl` |

The learning rate is 1e-4. Stages 1 and 2 took 20 and 18 hours on two NVIDIA RTX 4090 GPUs with a batch size of 4. The vocal units are the K = 250 clusters of layer-11 features of the stage-2 model. `transcribe.py` writes one label array per segment; these are released, merged with their timestamps, in `transcriptions_km250.jsonl.gz`. `trim_checkpoint.py` removes the optimizer state for release; the trimmed checkpoint gives identical features.

### 3. Vocal unit refinement

```python
from phoneme_finder import PhonemeFinder

finder = PhonemeFinder(parse_mode="energy",
                       parsed_transcription_file_path="metadata_trans.json",
                       valid_word_cnt_threshold=5,   # K: drop patterns seen fewer than 5 times
                       percentile=5)                 # merge unit pairs below the 5th-percentile JS divergence
finder.find_phonemes(context_mode="phoneme", num_context_phoneme=3, phoneme_merge_threshold=0.0)
```

- The input `metadata_trans.json` is written by `word_finder_energy.py` with ρ = 3 (`--threshold_coeffcient 3`). It holds the same fields as the released `transcriptions_km250.jsonl.gz`.
- We tried K ∈ {1, 3, 5} and the 5th, 10th and 15th percentiles, and selected K = 5 and the 5th percentile.
- Each run writes `config.json` (the cognitive vocal units and vocal patterns) and `mapping_phoneme.json` (the merged units). The paper's result is 105 cognitive vocal units and 69 vocal patterns.

## Reproducing the evaluation

| Result | How | Status |
|---|---|---|
| Table 2, dataset | `python evaluation/dataset_table2.py data` | Exact: 6,235 h of video, 152 h of vocalizations, 306,233 segments |
| SED F1 0.8556 | `1_preprocessing/sed/inference/test.py` on `beats_sed_dog.ckpt` | F1 0.8556 |
| Table 4, ABX | `evaluation/abx/phone_eval.py` | Answer keys for the 500 initial-unit and 210 final-unit triplets are in `evaluation/abx/samples/`. |
| Table 6, bark type | `evaluation/downstream/dt_aduio_classification.py` (`hubert_feature_ml_method_single`) | See Table 6 of the paper. |
| Table 7, bark context | `evaluation/downstream/dt_activity_classification.py` (`activity_classification_ml`) | Output in `evaluation/downstream/results/context/` (`f1-score`, macro avg). Labels: 1,008 valid clips annotated by five people. |
| Figure 5, Eq. 1 | `evaluation/vocal_patterns/word_meaning.py` | Relative frequencies of the 10 environment and 10 activity categories for every pattern, in `evaluation/vocal_patterns/results/relative_frequency.json`. |
| Figure 6, Zipf | `word_meaning.py::plot_word_distribution2` | Top 200 vocal patterns; the paper's figure is `evaluation/vocal_patterns/results/figure6_zipf.png`. |
| Table 11, the 69 patterns | `results/vocal_patterns_69_durations.json`, `evaluation/vocal_patterns/results/vocal_pattern_summaries.json` | Durations and descriptions match Table 11. |

## Citation

```bibtex
@inproceedings{wang-etal-2025-toward,
    title = "Toward Automatic Discovery of a Canine Phonetic Alphabet",
    author = "Wang, Theron S.  and Li, Xingyuan  and Lekhak, Hridayesh  and Dang, Tuan M.  and Wu, Mengyue  and Zhu, Kenny Q.",
    booktitle = "Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = jul,
    year = "2025",
    address = "Vienna, Austria",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.acl-long.451/",
    doi = "10.18653/v1/2025.acl-long.451",
    pages = "9207--9219",
}
```

## Acknowledgements

This code builds on PANNs, AudioSep, ATST-SED / DCASE 2023 Task 4, BEATs, fairseq HuBERT and auditok.
