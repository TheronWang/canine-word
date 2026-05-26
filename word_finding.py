# -*- coding: utf-8 -*-
# Authors: Theron S. Wang
# Date: 2025-12-29
# License: MIT
"""
Weakly Supervised Lexical Discovery for Phone Sequences

This module implements multiple methods for discovering "words" (meaningful phone subsequences)
from transcription data with weak keyword supervision.

Methods:
    1. N-Gram PMI: Statistical association with segmentation enumeration
    2. Boundary Model (EM): Learn boundary probabilities with optional keyword conditioning
    3. PrefixSpan (IG): Sequential pattern mining with Information Gain
    4. Neural Attention-MIL: CNN with attention (TODO)
    5. Adaptor Grammars: Bayesian hierarchical segmentation (TODO)
"""

from dataclasses import dataclass
import os
import json
import logging
import subprocess
from collections import defaultdict
from typing import List, Tuple, Dict, Optional
from unittest import result
import numpy as np
import shutil
import random
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib
import librosa
from abc import ABC, abstractmethod

matplotlib.use("Agg")  # Use non-interactive backend

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ==================== Model Data Classes ====================
@dataclass
class ParseResult:
    ngram_index: Dict[Tuple[int, ...], Dict[str, int]]
    keyword_counts: Dict[str, int]
    ngram_keyword_video_paths: Dict[Tuple[int, ...], Dict[str, set]]
    total_ngrams: int
    total_segments: int
    boundary_probs: Optional[Dict] = None  # Only for BoundaryModelParser

    def to_dict(self) -> Dict:
        """
        Convert ParseResult to JSON-serializable dictionary.

        Converts tuple keys to strings and sets to lists for JSON compatibility.

        :return: Dictionary with serializable values
        """
        return {
            "ngram_index": {
                str(list(ngram)): counts
                for ngram, counts in self.ngram_index.items()
            },
            "keyword_counts": self.keyword_counts,
            "ngram_keyword_video_paths": {
                str(list(ngram)): {
                    keyword: list(paths)
                    for keyword, paths in keyword_dict.items()
                }
                for ngram, keyword_dict in (
                    self.ngram_keyword_video_paths.items()
                )
            },
            "total_ngrams": self.total_ngrams,
            "total_segments": self.total_segments,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "ParseResult":
        """
        Create ParseResult from dictionary.

        Converts string keys back to tuples and lists back to sets.

        :param data: Dictionary created by to_dict()
        :return: ParseResult instance
        """
        import ast

        # Convert string keys back to tuple keys
        ngram_index = {
            tuple(ast.literal_eval(ngram_str)): counts
            for ngram_str, counts in data["ngram_index"].items()
        }

        ngram_keyword_video_paths = {
            tuple(ast.literal_eval(ngram_str)): {
                keyword: set(paths) for keyword, paths in keyword_dict.items()
            }
            for ngram_str, keyword_dict in data[
                "ngram_keyword_video_paths"
            ].items()
        }

        boundary_probs = None
        if "boundary_probs" in data:
            boundary_probs = {
                ast.literal_eval(k): v
                for k, v in data["boundary_probs"].items()
            }

        return cls(
            ngram_index=ngram_index,
            keyword_counts=data["keyword_counts"],
            ngram_keyword_video_paths=ngram_keyword_video_paths,
            total_ngrams=data["total_ngrams"],
            total_segments=data["total_segments"],
            boundary_probs=boundary_probs,
        )

    def save(self, filepath: str):
        """
        Save ParseResult to JSON file.

        :param filepath: Path to save the JSON file
        """
        import json

        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

        logger.info(f"Saved ParseResult to {filepath}")

    @classmethod
    def load(cls, filepath: str) -> "ParseResult":
        """
        Load ParseResult from JSON file.

        :param filepath: Path to the JSON file
        :return: ParseResult instance
        """
        import json

        with open(filepath, "r") as f:
            data = json.load(f)

        logger.info(f"Loaded ParseResult from {filepath}")
        return cls.from_dict(data)


# ==================== Abstract Base Classes ====================


class PhoneSequenceParser(ABC):
    """
    Abstract base class for phone sequence parsing strategies.

    A parser is responsible for segmenting phone sequences into n-grams and
    updating the parsed results in Sentence objects.
    """

    def __init__(self, ngram_range: Tuple[int, int]):
        """
        Initialize parser with n-gram range.

        :param ngram_range: (min_n, max_n) for N-gram extraction
        """
        self.ngram_range = ngram_range
        self.min_n, self.max_n = ngram_range

        # Shared data structures for n-gram metadata
        self.ngram_index = defaultdict(
            lambda: defaultdict(int)
        )  # {ngram: {keyword: count, 'total': count}}
        self.keyword_counts = defaultdict(int)  # {keyword: num_segments}
        self.ngram_keyword_video_paths = defaultdict(
            lambda: defaultdict(set)
        )  # {ngram: {keyword: set(video_paths)}}
        self.total_ngrams = 0
        self.total_segments = 0

    @abstractmethod
    def parse_sentences(self, sentences: Dict[str, "Sentence"]) -> ParseResult:
        """
        Parse all sentences and update their parsed results.

        This method should:
        1. Segment each sentence's segments into n-grams
        2. Update sentence.parsed_result, sentence.parsed_result_ngram_durations,
           sentence.parsed_result_ngram_ts
        3. Calculate n-gram metadata (counts, co-occurrences, etc.)

        :param sentences: Dictionary of {audio_name: Sentence}
        :return: Dictionary containing n-gram metadata
        """
        pass

    def _calculate_ngram_metadata(self, sentences: Dict[str, "Sentence"]):
        """
        Calculate n-gram counts and co-occurrences with keywords.

        Should be called after parsed_result is populated for all sentences.

        :param sentences: Dictionary of {audio_name: Sentence}
        """
        self.total_segments = sum(len(s.segments) for s in sentences.values())

        # Reset counters
        self.ngram_index.clear()
        self.keyword_counts.clear()
        self.ngram_keyword_video_paths.clear()
        self.total_ngrams = 0


class WordEvaluator(ABC):
    """
    Abstract base class for n-gram-keyword association evaluation.

    An evaluator computes scores measuring how strongly n-grams are
    associated with specific keywords.
    """

    def __init__(self, min_score: float = 0.0, min_support: int = 1):
        """
        Initialize evaluator with minimum score threshold.

        :param min_score: Minimum score for filtering results
        :param min_support: Minimum support count for filtering results
        """
        self.min_score = min_score
        self.min_support = min_support

    @abstractmethod
    def calculate_score(self, ngram: Tuple[int, ...], keyword: str) -> float:
        """
        Calculate association score between n-gram and keyword.

        :param ngram: Tuple of phone IDs
        :param keyword: Keyword string
        :return: Association score (higher = stronger association)
        """
        pass

    @abstractmethod
    def prepare_data(self, parse_result: ParseResult):
        """
        Prepare internal data structures from parser results.

        :param parse_result: ParseResult instance from parser
        """
        pass


class Sentence:
    """
    Represents a sentence with its transcription, keywords, and associated video.
    """

    DELIMITER = -1  # Frames without dog barks (segment delimiter)

    def __init__(
        self,
        audio_name: str,
        raw_transcription: List[int],
        audio_path: str,
        duration: float,
        keywords: set[str],
        video_path: Optional[str] = None,
        smooth: Optional[float] = None,
        length_prominent: int = 2,
        double_side: bool = False,
        be_replaced_length: int = 1,
    ):
        self.audio_name = audio_name
        self.raw_transcription = raw_transcription
        self.transcription = []  # Combined transcription
        self.total_frames = len(raw_transcription)
        self.phone_duration_transcription = []  # [duration_1, duration_2, ...]
        self.phone_ts_transcription = (
            []
        )  # [(start_1, end_1), (start_2, end_2), ...]
        self.segments = []  # List of segments after segmentation
        self.segments_durations = []  # [[duration_1, duration_2, ...], ...]
        self.segments_timestamps = (
            []
        )  # [[(start_1, end_1), (start_2, end_2), ...], ...]
        self.duration = duration
        self.keywords = keywords
        self.audio_path = audio_path
        self.video_path = video_path
        self.smooth = smooth
        if smooth is not None:
            self.length_prominent = length_prominent
            self.double_side = double_side
            self.be_replaced_length = be_replaced_length
        self.parsed_result = []  # [[(2,3), (5,7)], ... ] parsed segments
        self.parsed_result_ngram_durations = (
            []
        )  # [[duration_1, duration_2, ...], ... ]
        self.parsed_result_ngram_ts = (
            []
        )  # [[(start_1, end_1), (start_2, end_2), ...], ... ]

        if smooth is not None:
            self._smooth_transcription()
        self._combine_continous_identical_phones()
        self._segment_by_delimiter()

    def __repr__(self):
        return (
            f"Sentence(audio_name={self.audio_name}, keywords={self.keywords},"
            f" duration={self.duration})"
        )

    def _smooth_transcription(self):
        """
        Smooth the raw transcription by replacing isolated phones with surrounding prominent phones.

        Applies smoothing iteratively until no more changes can be made.
        Boundaries (start/end of transcription) are treated as always prominent.

        Parameters controlled by instance variables:
        - length_prominent: Number of continuous identical phones needed to be considered "prominent"
        - double_side: If True, both left and right sides must have the same prominent phone
        - be_replaced_length: Maximum length of consecutive phones to be replaced

        Example:
            [2, 2, 2, 19, 2, 2, 2] with length_prominent=3, double_side=True, be_replaced_length=1
            -> [2, 2, 2, 2, 2, 2, 2] (19 is replaced by 2)
        """
        if not self.raw_transcription or len(self.raw_transcription) == 0:
            return

        # Apply smoothing iteratively until no more changes
        max_iterations = 100  # Safety limit to prevent infinite loops
        iteration = 0

        while iteration < max_iterations:
            original = list(self.raw_transcription)
            smoothed = self._smooth_transcription_single_pass(original)

            # Check if any changes were made
            if smoothed == original:
                break

            self.raw_transcription = smoothed
            iteration += 1

    def _smooth_transcription_single_pass(self, original):
        """
        Perform a single pass of smoothing on the transcription.

        :param original: The transcription to smooth
        :return: Smoothed transcription
        """
        smoothed = list(original)
        n = len(original)
        i = 0

        while i < n:
            # Find the run of identical phones starting at position i in ORIGINAL
            current_phone = original[i]
            run_start = i
            run_end = i

            # Find the end of this run in ORIGINAL
            while run_end < n and original[run_end] == current_phone:
                run_end += 1

            run_length = run_end - run_start

            # Check if this run should be replaced
            # Only consider runs that are <= be_replaced_length
            if run_length <= self.be_replaced_length:
                # Check left side for prominent phone in ORIGINAL
                left_prominent_phone = None
                left_is_prominent = False

                if run_start == 0:
                    # At start boundary - not considered prominent from left
                    pass
                elif run_start >= self.length_prominent:
                    # Check if the phones before this run are all the same in ORIGINAL
                    left_candidate = original[run_start - 1]
                    is_prominent = True
                    for j in range(
                        run_start - self.length_prominent, run_start
                    ):
                        if original[j] != left_candidate:
                            is_prominent = False
                            break

                    if is_prominent:
                        left_prominent_phone = left_candidate
                        left_is_prominent = True

                # Check right side for prominent phone in ORIGINAL
                right_prominent_phone = None
                right_is_prominent = False

                if run_end == n:
                    # At end boundary - not considered prominent from right
                    pass
                elif run_end + self.length_prominent <= n:
                    # Check if the phones after this run are all the same in ORIGINAL
                    right_candidate = original[run_end]
                    is_prominent = True
                    for j in range(run_end, run_end + self.length_prominent):
                        if original[j] != right_candidate:
                            is_prominent = False
                            break

                    if is_prominent:
                        right_prominent_phone = right_candidate
                        right_is_prominent = True

                # Decide whether to replace
                replacement_phone = None

                if self.double_side:
                    # Both sides must be prominent and have the same phone
                    if (
                        left_is_prominent
                        and right_is_prominent
                        and left_prominent_phone == right_prominent_phone
                    ):
                        replacement_phone = left_prominent_phone
                else:
                    # One side must be prominent, and the other side must have the same phone
                    if left_is_prominent and right_is_prominent:
                        # Both sides are prominent, prefer if they match
                        if left_prominent_phone == right_prominent_phone:
                            replacement_phone = left_prominent_phone
                    elif left_is_prominent:
                        # Left is prominent, check if right side has the same phone (or at boundary)
                        if run_end < n:
                            right_phone = original[run_end]
                            if right_phone == left_prominent_phone:
                                replacement_phone = left_prominent_phone
                        else:
                            # At end boundary - allow replacement
                            replacement_phone = left_prominent_phone
                    elif right_is_prominent:
                        # Right is prominent, check if left side has the same phone (or at boundary)
                        if run_start > 0:
                            left_phone = original[run_start - 1]
                            if left_phone == right_prominent_phone:
                                replacement_phone = right_prominent_phone
                        else:
                            # At start boundary - allow replacement
                            replacement_phone = right_prominent_phone

                # Perform replacement in smoothed copy
                if replacement_phone is not None:
                    for j in range(run_start, run_end):
                        smoothed[j] = replacement_phone

            # Move to the next run (based on ORIGINAL)
            i = run_end

        return smoothed

    def _combine_continous_identical_phones(self):
        """
        Combine consecutive identical phones in the transcription.
        """
        if not self.raw_transcription:
            return
        combined_transcription = []
        last_phone = self.raw_transcription[0]
        phone_start_idx = 0

        for idx, phone in enumerate(self.raw_transcription[1:], start=1):
            if phone != last_phone:
                phone_end_idx = idx
                self.phone_duration_transcription.append(
                    self.duration
                    * (phone_end_idx - phone_start_idx)
                    / self.total_frames
                )
                self.phone_ts_transcription.append((
                    self.duration * phone_start_idx / self.total_frames,
                    self.duration * phone_end_idx / self.total_frames,
                ))
                combined_transcription.append(last_phone)
                phone_start_idx = idx
                last_phone = phone

        combined_transcription.append(last_phone)
        phone_end_idx = len(self.raw_transcription)
        self.phone_duration_transcription.append(
            self.duration
            * (phone_end_idx - phone_start_idx)
            / self.total_frames
        )
        self.phone_ts_transcription.append((
            self.duration * phone_start_idx / self.total_frames,
            self.duration * phone_end_idx / self.total_frames,
        ))
        self.transcription = combined_transcription

    def _segment_by_delimiter(self):
        """
        Segment combined transcription by delimiter (255).
        Also compute segment durations and timestamps.
        """
        segments = []
        segments_durations = []
        segments_timestamps = []

        current_segment = []
        current_durations = []
        current_timestamps = []

        for idx, phone in enumerate(self.transcription):
            if phone == self.DELIMITER:
                if len(current_segment) > 0:
                    segments.append(current_segment)
                    segments_durations.append(current_durations)
                    segments_timestamps.append(current_timestamps)
                    current_segment = []
                    current_durations = []
                    current_timestamps = []
            else:
                current_segment.append(phone)
                current_durations.append(
                    self.phone_duration_transcription[idx]
                )
                current_timestamps.append(self.phone_ts_transcription[idx])

        if len(current_segment) > 0:
            segments.append(current_segment)
            segments_durations.append(current_durations)
            segments_timestamps.append(current_timestamps)

        self.segments = segments
        self.segments_durations = segments_durations
        self.segments_timestamps = segments_timestamps

    def reset_parsed_results(self):
        """
        Reset parsed results to empty lists.
        """
        self.parsed_result = []
        self.parsed_result_ngram_durations = []
        self.parsed_result_ngram_ts = []

    def get_ngram_timestamp(
        self, segment_idx: int, ngram_pos: int, ngram_len: int
    ) -> Tuple[float, float]:
        """
        Get timestamp for an ngram in a specific segment.

        :param segment_idx: Index of the segment
        :param ngram_pos: Position of ngram in the segment
        :param ngram_len: Length of the ngram
        :return: (start_time, end_time) in seconds
        """
        if segment_idx >= len(self.segments_timestamps):
            return (0.0, 0.0)

        segment_ts = self.segments_timestamps[segment_idx]

        if ngram_pos >= len(segment_ts):
            return (0.0, 0.0)

        start_time = segment_ts[ngram_pos][0]

        end_pos = min(ngram_pos + ngram_len - 1, len(segment_ts) - 1)
        end_time = segment_ts[end_pos][1]

        return (start_time, end_time)


# ==================== Sample Generation ====================


class SampleGenerator:
    """_summary_

    Generates audio samples for each found N-gram-keyword pair.
    Generates audio samples for each long phones.
    Generates audio samples for each minimal pairs.

    Returns:
        _type_: _description_
    """

    def __init__(
        self,
        output_dir: str,
        sentences: Dict[str, "Sentence"],
        sample_rate: int = 16000,
        num_comparison_phone: int = 5,
    ):
        self.output_dir = Path(output_dir)
        # Shuffle the dictionary order
        sentence_items = list(sentences.items())
        random.shuffle(sentence_items)
        self.sentences = dict(sentence_items)
        self.sample_rate = sample_rate
        self.min_duration = 0.2
        self.sample_number_per_pair = 5
        self.sample_number_per_pair_minimal = 5
        self.sample_number_per_phone = 5
        self.num_comparison_phone = num_comparison_phone

    def generate_ngram_keyword_samples(
        self, parse_result: ParseResult, max_samples_per_pair: int = None
    ):
        """
        Generate audio samples for each n-gram-keyword pair found in parse results.

        :param parse_result: ParseResult instance from parser
        :param max_samples_per_pair: Maximum samples per n-gram-keyword pair (default: self.sample_number_per_pair)
        """
        if max_samples_per_pair is None:
            max_samples_per_pair = self.sample_number_per_pair

        output_dir = self.output_dir / "ngram_keyword_samples"
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Generating n-gram-keyword samples to {output_dir}")

        for (
            ngram,
            keyword_video_paths,
        ) in parse_result.ngram_keyword_video_paths.items():
            for keyword, video_paths in keyword_video_paths.items():
                # Create directory for this pair
                ngram_str = "_".join(map(str, ngram))
                pair_dir = output_dir / f"{ngram_str}_{keyword}"
                pair_dir.mkdir(parents=True, exist_ok=True)

                # Collect all instances of this n-gram-keyword pair
                instances = []
                for sentence in self.sentences.values():
                    if keyword not in sentence.keywords:
                        continue

                    for seg_idx, segment in enumerate(sentence.parsed_result):
                        for ngram_pos, found_ngram in enumerate(segment):
                            if found_ngram == ngram:
                                instances.append({
                                    "sentence_id": sentence.audio_name,
                                    "segment_idx": seg_idx,
                                    "ngram_pos": ngram_pos,
                                    "duration": (
                                        sentence.parsed_result_ngram_durations[
                                            seg_idx
                                        ][ngram_pos]
                                    ),
                                    "start_ts": (
                                        sentence.parsed_result_ngram_ts[
                                            seg_idx
                                        ][ngram_pos][0]
                                    ),
                                    "end_ts": sentence.parsed_result_ngram_ts[
                                        seg_idx
                                    ][ngram_pos][1],
                                    "audio_path": sentence.audio_path,
                                })

                # Randomly sample instances
                random.shuffle(instances)
                num_samples = min(max_samples_per_pair, len(instances))

                for i in range(num_samples):
                    sample = instances[i]
                    if sample["duration"] < self.min_duration:
                        continue

                    output_path = pair_dir / f"sample_{i+1}.wav"
                    self._extract_audio_segment(
                        sample["audio_path"],
                        sample["start_ts"],
                        sample["end_ts"],
                        output_path,
                    )

                logger.info(
                    f"Generated {num_samples} samples for {ngram_str} +"
                    f" {keyword}"
                )

    def generate_long_phone_samples(
        self,
        min_length: float = 0.5,
        max_samples_per_phone: int = None,
        generate_spectrograms: bool = False,
    ):
        """
        Generate audio samples for long phone instances.

        :param min_length: Minimum duration in seconds to consider a phone "long"
        :param max_samples_per_phone: Maximum samples per phone (default: self.sample_number_per_phone)
        :param generate_spectrograms: Whether to generate spectrograms for each sample
        """
        if max_samples_per_phone is None:
            max_samples_per_phone = self.sample_number_per_phone

        output_dir = self.output_dir / "long_phone_samples"
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            f"Generating long phone samples (min_length={min_length}s)"
        )

        # Collect long phone instances
        long_phones = defaultdict(list)

        for sentence in self.sentences.values():
            for phone_idx, (phone, duration) in enumerate(
                zip(
                    sentence.transcription,
                    sentence.phone_duration_transcription,
                )
            ):
                if phone == sentence.DELIMITER:
                    continue

                if duration >= min_length:
                    long_phones[phone].append({
                        "sentence_id": sentence.audio_name,
                        "phone_idx": phone_idx,
                        "duration": duration,
                        "start_ts": sentence.phone_ts_transcription[phone_idx][
                            0
                        ],
                        "end_ts": sentence.phone_ts_transcription[phone_idx][
                            1
                        ],
                        "audio_path": sentence.audio_path,
                    })

        # Generate samples for each phone
        for phone, instances in long_phones.items():
            phone_dir = output_dir / f"phone_{phone}"
            phone_dir.mkdir(parents=True, exist_ok=True)

            # Sort by duration (longest first)
            instances.sort(key=lambda x: x["duration"], reverse=True)

            num_samples = min(max_samples_per_phone, len(instances))

            audio_paths = []
            for i in range(num_samples):
                sample = instances[i]
                output_path = (
                    phone_dir
                    / f"sample_{i+1}_dur_{sample['duration']:.3f}s.wav"
                )
                self._extract_audio_segment(
                    sample["audio_path"],
                    sample["start_ts"],
                    sample["end_ts"],
                    output_path,
                )
                if generate_spectrograms:
                    audio_paths.append(output_path)

            # Generate combined spectrogram plot for this phone
            if generate_spectrograms and audio_paths:
                self._plot_spectrograms(
                    audio_paths, phone_dir / f"phone_{phone}_spectrograms.png"
                )

            logger.info(
                f"Generated {num_samples} samples for long phone {phone} (from"
                f" {len(instances)} instances)"
            )

    def generate_minimal_pair_samples(
        self,
        max_pairs: int = 100,
        generate_spectrograms: bool = False,
        pos_variance_phone: int = None,
        length_ngram: List[int] = None,
        num_sample: int = None,
    ):
        """
        Generate audio samples for minimal pairs (n-grams differing by exactly one phone).

        :param max_pairs: Maximum number of minimal pairs to generate samples for
        :param generate_spectrograms: Whether to generate spectrograms for each minimal pair
        :param pos_variance_phone: Position where the phone must differ (0-indexed), None for any position
        :param length_ngram: List of n-gram lengths to include, None for all lengths
        :param num_sample: Number of samples to generate for EACH ngram (ngram1 and ngram2), None uses default
        :return: Dictionary organized by differing phone: {phone_id: [pair_info_dicts]}
        """
        if num_sample is None:
            num_sample = self.sample_number_per_pair_minimal

        output_dir = self.output_dir / "minimal_pairs"
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Generating minimal pair samples...")
        logger.info(f"Number of samples per ngram: {num_sample}")
        if pos_variance_phone is not None:
            logger.info(
                f"Filtering for variance at position: {pos_variance_phone}"
            )
        if length_ngram is not None:
            # Handle both int and list inputs
            if isinstance(length_ngram, int):
                length_ngram = [length_ngram]
            logger.info(f"Filtering for n-gram lengths: {length_ngram}")

        found_ngram_ts = defaultdict(list)
        for sentence_id, sentence in self.sentences.items():
            for seg_idx, segment in enumerate(sentence.segments):
                if len(found_ngram_ts[tuple(segment)]) > 100:
                    pass
                found_ngram_ts[tuple(segment)].append({
                    "sentence_id": sentence_id,
                    "segment_idx": seg_idx,
                    "duration": sum(sentence.segments_durations[seg_idx]),
                    "start_ts": sentence.segments_timestamps[seg_idx][0][0],
                    "end_ts": sentence.segments_timestamps[seg_idx][-1][1],
                    "audio_path": (
                        sentence.audio_path
                        if sentence.audio_path.startswith("/redgiant/")
                        else "/redgiant/" + sentence.audio_path
                    ),
                })
        # for sentence_id, sentence in self.sentences.items():
        #     for seg_idx, segment in enumerate(sentence.parsed_result):
        #         for ngram_pos, ngram in enumerate(segment):
        #             if len(found_ngram_ts[ngram]) > 10000:
        #                 pass
        #             ngram_duration = sentence.parsed_result_ngram_durations[seg_idx][ngram_pos]
        #             found_ngram_ts[ngram].append({
        #                 "sentence_id": sentence_id,
        #                 "segment_idx": seg_idx,
        #                 "ngram_pos": ngram_pos,
        #                 "duration": ngram_duration,
        #                 "start_ts": sentence.parsed_result_ngram_ts[seg_idx][ngram_pos][0],
        #                 "end_ts": sentence.parsed_result_ngram_ts[seg_idx][ngram_pos][1],
        #                 "audio_path": (
        #                     sentence.audio_path
        #                     if sentence.audio_path.startswith("/redgiant/")
        #                     else "/redgiant/" + sentence.audio_path
        #                 ),
        #             })
        ngram_list = list(found_ngram_ts.keys())

        # Filter by length if specified
        if length_ngram is not None:
            ngram_list = [ng for ng in ngram_list if len(ng) in length_ngram]
            logger.info(f"After length filtering: {len(ngram_list)} n-grams")

        logger.info(
            f"Getting Minimal Pairs from {len(ngram_list)} found n-grams..."
        )
        pairs_by_phone_pair = self._get_minimal_pairs_efficient(
            ngram_list, pos_variance_phone
        )

        # Count total pairs
        total_pairs = sum(len(pairs) for pairs in pairs_by_phone_pair.values())
        logger.info(
            f"Found {total_pairs} minimal pairs organized by"
            f" {len(pairs_by_phone_pair)} phone pairs"
        )

        # Sample pairs if needed
        if total_pairs > max_pairs:
            # Collect all pairs and sample
            all_pairs_with_key = []
            for phone_pair_key, pairs in pairs_by_phone_pair.items():
                for pair in pairs:
                    all_pairs_with_key.append((phone_pair_key, pair))
            sampled = random.sample(all_pairs_with_key, max_pairs)
            # Re-organize by phone pair
            pairs_by_phone_pair = defaultdict(list)
            for phone_pair_key, pair in sampled:
                pairs_by_phone_pair[phone_pair_key].append(pair)
            pairs_by_phone_pair = dict(pairs_by_phone_pair)

        # Generate samples organized by phone pairs
        results_by_phone_pair = defaultdict(list)

        for phone_pair_key, pairs_list in pairs_by_phone_pair.items():
            for pair_info in pairs_list:
                ngram1 = pair_info["ngram1"]
                ngram2 = pair_info["ngram2"]
                diff_pos = pair_info["diff_pos"]
                diff_phone1 = pair_info["diff_phone1"]
                diff_phone2 = pair_info["diff_phone2"]

                pair_dir = (
                    output_dir
                    / f"phone_pair_{phone_pair_key}"
                    / f"{'_'.join(map(str, ngram1))}_vs_{'_'.join(map(str, ngram2))}"
                )
                pair_dir.mkdir(parents=True, exist_ok=True)

                samples_1 = found_ngram_ts[ngram1]
                samples_2 = found_ngram_ts[ngram2]

                random.shuffle(samples_1)
                random.shuffle(samples_2)

                audio_paths = []
                sample_info = {
                    "ngram1": ngram1,
                    "ngram2": ngram2,
                    "diff_pos": diff_pos,
                    "diff_phone1": diff_phone1,
                    "diff_phone2": diff_phone2,
                    "phone_pair_key": phone_pair_key,
                    "samples": [],
                }

            # Extract ngram1 samples, skipping those with insufficient duration
            samples_extracted = 0
            sample_idx = 0
            while samples_extracted < num_sample and sample_idx < len(
                samples_1
            ):
                sample = samples_1[sample_idx]
                sample_idx += 1
                if sample["duration"] < self.min_duration:
                    continue
                output_path = (
                    pair_dir / f"ngram1_sample_{samples_extracted+1}.wav"
                )
                self._extract_audio_segment(
                    sample["audio_path"],
                    sample["start_ts"],
                    sample["end_ts"],
                    output_path,
                )
                sample_info["samples"].append({
                    "ngram": "ngram1",
                    "audio_path": str(output_path),
                    "source": sample,
                })
                if generate_spectrograms:
                    audio_paths.append(output_path)
                samples_extracted += 1

            # Extract ngram2 samples, skipping those with insufficient duration
            samples_extracted = 0
            sample_idx = 0
            while samples_extracted < num_sample and sample_idx < len(
                samples_2
            ):
                sample = samples_2[sample_idx]
                sample_idx += 1
                if sample["duration"] < self.min_duration:
                    continue
                output_path = (
                    pair_dir / f"ngram2_sample_{samples_extracted+1}.wav"
                )
                self._extract_audio_segment(
                    sample["audio_path"],
                    sample["start_ts"],
                    sample["end_ts"],
                    output_path,
                )
                sample_info["samples"].append({
                    "ngram": "ngram2",
                    "audio_path": str(output_path),
                    "source": sample,
                })
                if generate_spectrograms:
                    audio_paths.append(output_path)
                samples_extracted += 1

                # Generate combined spectrogram plot for this minimal pair
                if generate_spectrograms and audio_paths:
                    pair_name = (
                        f"{'_'.join(map(str, ngram1))}_vs_{'_'.join(map(str, ngram2))}"
                    )
                    self._plot_spectrograms(
                        audio_paths, pair_dir / f"{pair_name}_spectrograms.png"
                    )

                results_by_phone_pair[phone_pair_key].append(sample_info)

        total_pairs_generated = sum(
            len(pairs) for pairs in results_by_phone_pair.values()
        )
        logger.info(
            f"Generated samples for {total_pairs_generated} minimal pairs"
        )
        logger.info(f"Organized by {len(results_by_phone_pair)} phone pairs")

        return dict(results_by_phone_pair)

    def _get_minimal_pairs_efficient(
        self, ngram_list: List[Tuple[int, ...]], pos_variance_phone: int = None
    ) -> Dict[str, List[Dict]]:
        """
        Efficiently find all minimal pairs in the ngram_list, organized by phone pairs.
        For each unique phone, randomly selects up to num_comparison_phone other phones to compare with.

        :param ngram_list: List of n-grams (tuples of phone IDs)
        :param pos_variance_phone: Position where the phone must differ (0-indexed), None for any position
        :return: Dict organized by phone pairs: {"phone1-phone2": [{"ngram1": ..., "ngram2": ..., "diff_pos": ..., "diff_phone1": ..., "diff_phone2": ...}, ...]}
        """
        ngram_by_length = defaultdict(list)
        for ngram in tqdm(ngram_list):
            ngram_by_length[len(ngram)].append(ngram)

        # randomly order the n-grams in each length group to diversify pairs
        for n in ngram_by_length:
            random.shuffle(ngram_by_length[n])

        # First, collect all pairs without organizing
        all_pairs = []
        for n in ngram_by_length:
            pairs = self._get_minimal_pairs(
                ngram_by_length[n][:10000], pos_variance_phone
            )
            all_pairs.extend(pairs)

        # Organize pairs by phone pairs with random sampling
        pairs_by_phone_pair = self._organize_pairs_by_phone_pairs(all_pairs)

        return pairs_by_phone_pair

    def _organize_pairs_by_phone_pairs(
        self, all_pairs: List[Dict]
    ) -> Dict[str, List[Dict]]:
        """
        Organize pairs by phone pairs with random sampling of comparison phones.

        :param all_pairs: List of all pair info dicts
        :return: Dict organized by phone pair keys
        """
        # Group pairs by individual phones to understand which phones compare with which
        phone_comparisons = defaultdict(
            set
        )  # {phone: set of other phones it compares with}

        for pair in all_pairs:
            phone1 = pair["diff_phone1"]
            phone2 = pair["diff_phone2"]
            phone_comparisons[phone1].add(phone2)
            phone_comparisons[phone2].add(phone1)

        # For each phone, randomly select up to num_comparison_phone other phones
        selected_phone_pairs = set()
        for phone, comparison_phones in phone_comparisons.items():
            comparison_list = list(comparison_phones)
            # Randomly sample up to num_comparison_phone
            num_to_select = min(
                self.num_comparison_phone, len(comparison_list)
            )
            selected_comparisons = random.sample(
                comparison_list, num_to_select
            )

            for other_phone in selected_comparisons:
                # Create a sorted pair to avoid duplicates like (1,2) and (2,1)
                phone_pair = tuple(sorted([phone, other_phone]))
                selected_phone_pairs.add(phone_pair)

        # Now organize pairs using selected phone pairs
        pairs_by_phone_pair = defaultdict(list)
        for pair in all_pairs:
            phone1 = pair["diff_phone1"]
            phone2 = pair["diff_phone2"]
            phone_pair = tuple(sorted([phone1, phone2]))

            if phone_pair in selected_phone_pairs:
                phone_pair_key = f"{phone_pair[0]}-{phone_pair[1]}"
                pairs_by_phone_pair[phone_pair_key].append(pair)

        logger.info(
            f"Selected {len(selected_phone_pairs)} phone pairs from"
            f" {len(phone_comparisons)} unique phones"
        )
        return dict(pairs_by_phone_pair)

    def _get_minimal_pairs(
        self, ngram_list: List[Tuple[int, ...]], pos_variance_phone: int = None
    ) -> List[Dict]:
        minimal_pairs = []
        n = len(ngram_list)
        for i in range(n):
            for j in range(i + 1, n):
                pair_info = self._is_minimal_pair(
                    ngram_list[i], ngram_list[j], pos_variance_phone
                )
                if pair_info:
                    minimal_pairs.append(pair_info)
        return minimal_pairs

    def _is_minimal_pair(
        self,
        ngram1: Tuple[int, ...],
        ngram2: Tuple[int, ...],
        pos_variance_phone: int = None,
    ) -> Optional[Dict]:
        """
        Check if two n-grams form a minimal pair and return pair information.

        :param ngram1: First n-gram
        :param ngram2: Second n-gram
        :param pos_variance_phone: Position where the phone must differ (0-indexed), None for any position
        :return: Dict with pair info if minimal pair, None otherwise
        """
        if len(ngram1) != len(ngram2):
            return None

        diff_positions = []
        for idx, (a, b) in enumerate(zip(ngram1, ngram2)):
            if a != b:
                diff_positions.append(idx)

        if len(diff_positions) != 1:
            return None

        diff_pos = diff_positions[0]

        # Check position constraint if specified
        if pos_variance_phone is not None and diff_pos != pos_variance_phone:
            return None

        return {
            "ngram1": ngram1,
            "ngram2": ngram2,
            "diff_pos": diff_pos,
            "diff_phone1": ngram1[diff_pos],
            "diff_phone2": ngram2[diff_pos],
        }

    def _extract_audio_segment(
        self,
        audio_path: str,
        start_ts: float,
        end_ts: float,
        output_path: Path,
    ):
        """
        Extract audio segment using ffmpeg.

        :param audio_path: Path to source audio file
        :param start_ts: Start timestamp in seconds
        :param end_ts: End timestamp in seconds
        :param output_path: Path to save extracted audio
        """
        import subprocess

        duration = end_ts - start_ts

        # FFmpeg command to extract audio segment
        cmd = [
            "ffmpeg",
            "-i",
            audio_path,
            "-ss",
            str(start_ts),
            "-t",
            str(duration),
            "-ar",
            str(self.sample_rate),
            "-ac",
            "1",  # mono
            "-y",  # overwrite output file
            str(output_path),
        ]

        try:
            subprocess.run(cmd, check=True, capture_output=True)
        except subprocess.CalledProcessError as e:
            logger.error(
                f"Failed to extract audio segment from {audio_path}"
                f" [{start_ts:.2f}s - {end_ts:.2f}s]: {e.stderr.decode()}"
            )
        except FileNotFoundError:
            logger.error(
                "ffmpeg not found. Please install ffmpeg to extract audio"
                " segments."
            )

    def _plot_spectrograms(self, audio_paths: List[Path], output_path: Path):
        """
        Plot spectrograms for multiple audio files in a single figure.

        :param audio_paths: List of paths to audio files
        :param output_path: Path to save the combined spectrogram plot
        """
        import librosa
        import librosa.display

        num_samples = len(audio_paths)
        if num_samples == 0:
            return

        # Create figure with subplots
        fig, axes = plt.subplots(num_samples, 1, figsize=(12, 3 * num_samples))
        if num_samples == 1:
            axes = [axes]

        for idx, audio_path in enumerate(audio_paths):
            try:
                # Load audio file
                y, sr = librosa.load(str(audio_path), sr=self.sample_rate)

                # Compute mel spectrogram
                mel_spec = librosa.feature.melspectrogram(
                    y=y, sr=sr, n_mels=128, fmax=8000
                )
                mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

                # Plot spectrogram
                img = librosa.display.specshow(
                    mel_spec_db,
                    sr=sr,
                    x_axis="time",
                    y_axis="mel",
                    fmax=8000,
                    ax=axes[idx],
                    cmap="viridis",
                )

                axes[idx].set_title(f"{audio_path.name}", fontsize=10)
                axes[idx].set_ylabel("Frequency (Hz)")

                # Add colorbar
                fig.colorbar(img, ax=axes[idx], format="%+2.0f dB")

            except Exception as e:
                logger.error(
                    f"Failed to generate spectrogram for {audio_path}: {e}"
                )
                axes[idx].text(
                    0.5,
                    0.5,
                    f"Error: {str(e)}",
                    ha="center",
                    va="center",
                    transform=axes[idx].transAxes,
                )

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved combined spectrograms to {output_path}")


# ==================== Concrete Parser Implementations ====================


class SlidingWindowParser(PhoneSequenceParser):
    """
    Sliding window parser that extracts all n-grams of sizes min_n to max_n.

    This parser uses a simple sliding window approach without any learning.
    Each position in a segment generates n-grams of all valid sizes.
    """

    def __init__(self, ngram_range: Tuple[int, int]):
        super().__init__(ngram_range)

    def parse_sentences(self, sentences: Dict[str, "Sentence"]) -> ParseResult:
        """
        Parse sentences using sliding window approach.

        Updates each sentence's parsed_result to contain all n-grams extracted
        via sliding window.

        :param sentences: Dictionary of {audio_name: Sentence}
        :return: ParseResult instance with metadata
        """
        logger.info("Parsing with Sliding Window approach...")
        logger.info(f"N-gram range: {self.min_n}-{self.max_n}")

        for sentence in sentences.values():
            sentence.reset_parsed_results()

            for seg_idx, segment in enumerate(sentence.segments):
                # For sliding window, we extract all n-grams as the "parsing"
                # This creates overlapping n-grams
                segment_ngrams = []
                segment_durations = []
                segment_timestamps = []

                # Extract all n-grams using sliding window
                for n in range(
                    self.min_n, min(self.max_n + 1, len(segment) + 1)
                ):
                    for i in range(len(segment) - n + 1):
                        ngram = tuple(segment[i : i + n])
                        segment_ngrams.append(ngram)

                        # Calculate duration and timestamp for this n-gram
                        duration = sum(
                            sentence.segments_durations[seg_idx][i : i + n]
                        )
                        segment_durations.append(duration)

                        start_time = sentence.segments_timestamps[seg_idx][i][
                            0
                        ]
                        end_time = sentence.segments_timestamps[seg_idx][
                            i + n - 1
                        ][1]
                        segment_timestamps.append((start_time, end_time))

                # Store parsed result for this segment
                sentence.parsed_result.append(segment_ngrams)
                sentence.parsed_result_ngram_durations.append(
                    segment_durations
                )
                sentence.parsed_result_ngram_ts.append(segment_timestamps)

        # Calculate metadata
        self._calculate_ngram_metadata(sentences)

        return ParseResult(
            ngram_index=dict(self.ngram_index),
            keyword_counts=dict(self.keyword_counts),
            ngram_keyword_video_paths=dict(self.ngram_keyword_video_paths),
            total_ngrams=self.total_ngrams,
            total_segments=self.total_segments,
        )

    def _calculate_ngram_metadata(self, sentences: Dict[str, "Sentence"]):
        """Calculate n-gram metadata for sliding window approach."""
        super()._calculate_ngram_metadata(sentences)

        for sentence in sentences.values():
            audio_keywords = sentence.keywords
            ngrams_in_audio = (
                set()
            )  # Track unique ngrams per audio for support counting

            for seg_idx, segment in enumerate(sentence.segments):
                # Update keyword counts (once per segment)
                for keyword in audio_keywords:
                    self.keyword_counts[keyword] += 1

                # Extract all n-grams using sliding window
                for n in range(
                    self.min_n, min(self.max_n + 1, len(segment) + 1)
                ):
                    for i in range(len(segment) - n + 1):
                        ngram = tuple(segment[i : i + n])

                        # Count occurrence
                        self.ngram_index[ngram]["total"] += 1
                        self.total_ngrams += 1

                        # Track for support counting
                        ngrams_in_audio.add(ngram)

                        # Track co-occurrence with keywords
                        for keyword in audio_keywords:
                            self.ngram_index[ngram][keyword] += 1

                        # Track video paths
                        if sentence.video_path:
                            for keyword in audio_keywords:
                                self.ngram_keyword_video_paths[ngram][
                                    keyword
                                ].add(sentence.video_path)

        logger.info(f"Total Unique N-grams: {len(self.ngram_index):,}")
        logger.info(f"Total N-gram instances: {self.total_ngrams:,}")


class BoundaryModelParser(PhoneSequenceParser):
    """
    EM-based boundary model parser that learns segmentation patterns.

    This parser learns P(boundary | context) or P(boundary | context, keyword)
    using an EM algorithm, then segments sequences based on learned boundaries.
    """

    # Boundary Model Parameters
    BOUNDARY_PRIOR = 0.1
    SEGMENT_PENALTY = -0.5
    USE_LENGTH_NORMALIZATION = True

    def __init__(
        self,
        ngram_range: Tuple[int, int],
        max_iterations: int = 5,
        smoothing: float = 0.1,
        use_keyword_context: bool = False,
        convergence_threshold: float = 1e-4,
        segment_penalty: float = None,
    ):
        super().__init__(ngram_range)
        self.max_iterations = max_iterations
        self.smoothing = smoothing
        self.use_keyword_context = use_keyword_context
        self.convergence_threshold = convergence_threshold
        self.segment_penalty = (
            segment_penalty
            if segment_penalty is not None
            else self.SEGMENT_PENALTY
        )
        self.boundary_probs = defaultdict(lambda: self.BOUNDARY_PRIOR)

    def parse_sentences(self, sentences: Dict[str, "Sentence"]) -> ParseResult:
        """
        Parse sentences using EM-based boundary model.

        Learns boundary probabilities and segments sequences accordingly.

        :param sentences: Dictionary of {audio_name: Sentence}
        :return: ParseResult instance with metadata including boundary_probs
        """
        logger.info("Parsing with Boundary Model (EM) approach...")
        logger.info(f"N-gram range: {self.min_n}-{self.max_n}")
        logger.info(f"Max iterations: {self.max_iterations}")
        logger.info(f"Keyword-conditional: {self.use_keyword_context}")
        logger.info(f"Length normalization: {self.USE_LENGTH_NORMALIZATION}")
        logger.info(f"Segment penalty: {self.segment_penalty}")

        # Initialize boundary probability structures
        if self.use_keyword_context:
            boundary_counts = defaultdict(int)
            context_counts = defaultdict(int)
            boundary_probs = defaultdict(lambda: self.BOUNDARY_PRIOR)
        else:
            boundary_counts = defaultdict(int)
            context_counts = defaultdict(int)
            boundary_probs = defaultdict(lambda: self.BOUNDARY_PRIOR)

        # EM iterations
        prev_avg_log_prob = -float("inf")

        for iteration in range(self.max_iterations):
            logger.info(f"\nIteration {iteration + 1}/{self.max_iterations}")

            # Reset counts for M-step
            boundary_counts.clear()
            context_counts.clear()

            total_log_prob = 0.0
            total_phones = 0

            # E-step: Find best segmentation for each segment
            for sentence in sentences.values():
                sentence.reset_parsed_results()
                keywords_tuple = (
                    tuple(sorted(sentence.keywords))
                    if self.use_keyword_context
                    else None
                )

                for seg_idx, segment in enumerate(sentence.segments):
                    if len(segment) == 0:
                        continue

                    # Find best segmentation
                    best_seg, log_prob, durations, timestamps = (
                        self._find_best_segmentation_boundary(
                            sentence,
                            seg_idx,
                            segment,
                            self.min_n,
                            self.max_n,
                            boundary_probs,
                            (
                                list(sentence.keywords)
                                if keywords_tuple
                                else None
                            ),
                        )
                    )

                    # Store parsed result
                    sentence.parsed_result.append(best_seg)
                    sentence.parsed_result_ngram_durations.append(durations)
                    sentence.parsed_result_ngram_ts.append(timestamps)

                    # Update statistics
                    total_log_prob += log_prob
                    total_phones += len(segment)

                    # M-step: Update boundary counts
                    self._update_boundary_counts(
                        segment,
                        best_seg,
                        boundary_counts,
                        context_counts,
                        max_context=3,
                        keywords=keywords_tuple,
                    )

            # M-step: Update boundary probabilities
            for context in context_counts:
                count = context_counts[context]
                if count > 0:
                    boundary_probs[context] = (
                        boundary_counts[context] + self.smoothing
                    ) / (count + 2 * self.smoothing)

            # Check convergence
            avg_log_prob = total_log_prob / max(total_phones, 1)
            improvement = avg_log_prob - prev_avg_log_prob
            logger.info(
                f"Average log-prob: {avg_log_prob:.6f}, Improvement:"
                f" {improvement:.6f}"
            )

            if abs(improvement) < self.convergence_threshold:
                logger.info(f"Converged after {iteration + 1} iterations")
                break

            prev_avg_log_prob = avg_log_prob

        # Store final boundary model
        self.boundary_probs = boundary_probs

        # Calculate metadata
        self._calculate_ngram_metadata(sentences)

        return ParseResult(
            ngram_index=dict(self.ngram_index),
            keyword_counts=dict(self.keyword_counts),
            ngram_keyword_video_paths=dict(self.ngram_keyword_video_paths),
            total_ngrams=self.total_ngrams,
            total_segments=self.total_segments,
            boundary_probs=dict(self.boundary_probs),
        )

    def _calculate_ngram_metadata(self, sentences: Dict[str, "Sentence"]):
        """Calculate n-gram metadata from parsed results."""
        super()._calculate_ngram_metadata(sentences)

        for sentence in sentences.values():
            audio_keywords = sentence.keywords

            for seg_idx, parsed_segment in enumerate(sentence.parsed_result):
                # Update keyword counts (once per segment)
                for keyword in audio_keywords:
                    self.keyword_counts[keyword] += 1

                # Count n-grams from parsed segmentation
                for ngram in parsed_segment:
                    self.ngram_index[ngram]["total"] += 1
                    self.total_ngrams += 1

                    # Track co-occurrence with keywords
                    for keyword in audio_keywords:
                        self.ngram_index[ngram][keyword] += 1

                    # Track video paths
                    if sentence.video_path:
                        for keyword in audio_keywords:
                            self.ngram_keyword_video_paths[ngram][keyword].add(
                                sentence.video_path
                            )

        logger.info(f"Total Unique N-grams: {len(self.ngram_index):,}")
        logger.info(f"Total N-gram instances: {self.total_ngrams:,}")
        logger.info(f"Keywords: {list(self.keyword_counts.keys())}")

    def _get_boundary_context(
        self, segment: List[int], position: int, max_context: int = 3
    ) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
        """Extract left and right context for a boundary position."""
        left_start = max(0, position + 1 - max_context)
        left_context = tuple(segment[left_start : position + 1])

        right_end = min(len(segment), position + 1 + max_context)
        right_context = tuple(segment[position + 1 : right_end])

        return (left_context, right_context)

    def _find_best_segmentation_boundary(
        self,
        sentence: "Sentence",
        seg_idx: int,
        segment: List[int],
        min_n: int,
        max_n: int,
        boundary_probs: Dict,
        keywords: List[str] = None,
    ) -> Tuple[
        List[Tuple[int, ...]], float, List[float], List[Tuple[float, float]]
    ]:
        """Find best segmentation using boundary probabilities via DP."""
        n = len(segment)
        if n == 0:
            return ([], 0.0, [], [])

        # Pre-calculate boundary probability matrix
        P_boundary = []
        for pos in range(n - 1):
            left_ctx, right_ctx = self._get_boundary_context(
                segment, pos, max_context=3
            )
            p_bound = self._get_boundary_prob(
                boundary_probs,
                left_ctx,
                right_ctx,
                tuple(keywords) if keywords else None,
            )
            P_boundary.append(p_bound)

        # Compute prefix sum for "no boundary" probabilities
        prefix_log_no_boundary = [0.0]
        for i in range(len(P_boundary)):
            prefix_log_no_boundary.append(
                prefix_log_no_boundary[-1]
                + np.log(max(1 - P_boundary[i], 1e-10))
            )

        # DP: dp[i] = (cumulative_log_prob, segmentation, durations, timestamps, num_segments)
        dp = [(-float("inf"), [], [], [], 0)] * (n + 1)
        dp[0] = (0.0, [], [], [], 0)

        for i in range(1, n + 1):
            for length in range(min_n, min(max_n, i) + 1):
                j = i - length
                if dp[j][0] == -float("inf"):
                    continue

                # Calculate score for adding segment [j:i]
                prev_score = dp[j][0]
                prev_num_segs = dp[j][4]

                # Log probability of no boundaries within [j, i)
                log_no_boundary_within = (
                    prefix_log_no_boundary[i - 1] - prefix_log_no_boundary[j]
                )

                # Log probability of boundary at position i-1 (if not at end)
                log_boundary_at_i = 0.0
                if i < n:
                    log_boundary_at_i = np.log(max(P_boundary[i - 1], 1e-10))

                # Total score for this segment
                segment_score = (
                    log_no_boundary_within
                    + log_boundary_at_i
                    + self.segment_penalty
                )

                # Calculate cumulative score
                cumulative_score = prev_score + segment_score
                new_num_segs = prev_num_segs + 1

                # Normalize for comparison if enabled
                if self.USE_LENGTH_NORMALIZATION:
                    normalized_score = cumulative_score / i
                    prev_normalized = (
                        dp[i][0] / i
                        if dp[i][0] != -float("inf")
                        else -float("inf")
                    )
                    is_better = normalized_score > prev_normalized
                else:
                    is_better = cumulative_score > dp[i][0]

                if is_better:
                    ngram = tuple(segment[j:i])
                    duration = sum(sentence.segments_durations[seg_idx][j:i])
                    start_time = sentence.segments_timestamps[seg_idx][j][0]
                    end_time = sentence.segments_timestamps[seg_idx][i - 1][1]

                    dp[i] = (
                        cumulative_score,
                        dp[j][1] + [ngram],
                        dp[j][2] + [duration],
                        dp[j][3] + [(start_time, end_time)],
                        new_num_segs,
                    )

        # Return final result
        final_cumulative_score = dp[n][0]
        if self.USE_LENGTH_NORMALIZATION and n > 0:
            final_normalized_score = final_cumulative_score / n
        else:
            final_normalized_score = final_cumulative_score

        return (
            dp[n][1] if dp[n][1] else [],
            final_normalized_score,
            dp[n][2],
            dp[n][3],
        )

    def _get_boundary_prob(
        self,
        boundary_probs: Dict,
        left_ctx: Tuple,
        right_ctx: Tuple,
        keywords: Tuple[str, ...] = None,
    ) -> float:
        """Get boundary probability."""
        if keywords:
            # Average over all keywords
            probs = []
            for kw in keywords:
                key = (left_ctx, right_ctx, kw)
                probs.append(boundary_probs.get(key, self.BOUNDARY_PRIOR))
            return np.mean(probs) if probs else self.BOUNDARY_PRIOR
        else:
            key = (left_ctx, right_ctx)
            return boundary_probs.get(key, self.BOUNDARY_PRIOR)

    def _update_boundary_counts(
        self,
        segment: List[int],
        best_segmentation: List[Tuple[int, ...]],
        boundary_counts: Dict,
        context_counts: Dict,
        max_context: int = 3,
        keywords: Tuple[str, ...] = None,
    ):
        """Update boundary statistics based on chosen segmentation."""
        # Determine boundary positions
        boundary_positions = set()
        pos = 0
        for ngram in best_segmentation[:-1]:
            pos += len(ngram)
            boundary_positions.add(pos - 1)

        # Update counts for all possible positions
        for pos in range(len(segment) - 1):
            left_ctx, right_ctx = self._get_boundary_context(
                segment, pos, max_context
            )
            is_boundary = pos in boundary_positions

            if keywords:
                for kw in keywords:
                    key = (left_ctx, right_ctx, kw)
                    context_counts[key] += 1
                    if is_boundary:
                        boundary_counts[key] += 1
            else:
                key = (left_ctx, right_ctx)
                context_counts[key] += 1
                if is_boundary:
                    boundary_counts[key] += 1


# ==================== Concrete Evaluator Implementations ====================


class NPMIEvaluator(WordEvaluator):
    """
    Normalized Pointwise Mutual Information evaluator.

    Calculates NPMI score measuring statistical association between
    n-grams and keywords.
    """

    def __init__(self, min_score: float = 0.2, min_support: int = 50):
        super().__init__(min_score, min_support)
        self.ngram_index = {}
        self.keyword_counts = {}
        self.total_ngrams = 0
        self.total_segments = 0

        self.valid_ngram_keyword_pairs_list = []

    def prepare_data(self, parse_result: ParseResult):
        """
        Prepare data structures from parser results.

        :param parse_result: ParseResult instance from parser
        :return: None
        """
        self.ngram_index = parse_result.ngram_index
        self.keyword_counts = parse_result.keyword_counts
        self.total_ngrams = parse_result.total_ngrams
        self.total_segments = parse_result.total_segments

        logger.info(
            f"NPMI Evaluator prepared with {len(self.ngram_index)} n-grams"
        )

    def calculate_score(self, ngram: Tuple[int, ...], keyword: str) -> float:
        """
        Calculate Normalized Pointwise Mutual Information (NPMI).

        Formula:
            PMI = log₂(P(w,k) / (P(w) × P(k)))
            NPMI = PMI / -log₂(P(w,k))

        :param ngram: Tuple of phone IDs
        :param keyword: Keyword string
        :return: NPMI score in [-1, 1]
        """
        if ngram not in self.ngram_index:
            return -1.0

        # P(w) - probability of N-gram
        p_w = self.ngram_index[ngram]["total"] / self.total_ngrams

        # P(k) - probability of keyword
        p_k = self.keyword_counts[keyword] / self.total_segments

        # P(w, k) - joint probability
        count_w_k = self.ngram_index[ngram].get(keyword, 0)
        if count_w_k == 0:
            return -1.0

        p_w_k = count_w_k / self.total_ngrams

        # Avoid log(0)
        if p_w_k <= 0 or p_w <= 0 or p_k <= 0:
            return -1.0

        # PMI calculation
        pmi = np.log2(p_w_k / (p_w * p_k))

        # Normalized PMI
        npmi = pmi / -np.log2(p_w_k)

        return npmi

    def find_valid_ngram_keyword_pairs(
        self, result_folder: str = None
    ) -> List[Dict]:
        """
        Find ngram-keyword pairs exceeding thresholds both for support and NPMI.

        :param min_support: Minimum frequency of n-gram for consideration
        :param min_npmi: Minimum NPMI score for consideration
        :param result_folder: Optional folder path to save JSON results
        :return: List of dicts with keys: ngram, keyword, support, npmi
        """
        if not self.ngram_index:
            logger.warning(
                "NPMI Evaluator data not prepared. Call prepare_data() first."
            )
            return []

        # Reset list to avoid accumulation across multiple calls
        self.valid_ngram_keyword_pairs_list = []

        for ngram, counts in self.ngram_index.items():
            for keyword in counts:
                if keyword == "total":
                    continue
                if counts[keyword] >= self.min_support:
                    npmi = self.calculate_score(ngram, keyword)
                    if npmi >= self.min_score:
                        self.valid_ngram_keyword_pairs_list.append({
                            "ngram": list(
                                ngram
                            ),  # Convert tuple to list for JSON serialization
                            "keyword": keyword,
                            "npmi": float(npmi),  # Ensure JSON serializable
                            "support": counts[keyword],
                            "ngram_cnt": counts["total"],
                            "keyword_cnt": self.keyword_counts[keyword],
                        })

        logger.info(
            f"Found {len(self.valid_ngram_keyword_pairs_list)} valid"
            " ngram-keyword pairs"
        )

        if result_folder:
            os.makedirs(result_folder, exist_ok=True)

            # save config of this evaluator
            config_path = os.path.join(
                result_folder, "npmi_evaluator_config.json"
            )
            with open(config_path, "w") as f:
                json.dump(
                    {
                        "min_score": self.min_score,
                        "min_support": self.min_support,
                    },
                    f,
                    indent=4,
                )
            logger.info(f"Saved NPMI evaluator config to {config_path}")

            # save results to a json file, order by npmi descending
            result_path = os.path.join(
                result_folder, "valid_ngram_keyword_pairs_by_npmi.json"
            )
            with open(result_path, "w") as f:
                json.dump(
                    sorted(
                        self.valid_ngram_keyword_pairs_list,
                        key=lambda x: x["npmi"],
                        reverse=True,
                    ),
                    f,
                    indent=4,
                )
            logger.info(
                f"Saved valid ngram-keyword pairs (by NPMI) to {result_path}"
            )

            # save results to a json file, order by support descending
            result_path = os.path.join(
                result_folder, "valid_ngram_keyword_pairs_by_support.json"
            )
            with open(result_path, "w") as f:
                json.dump(
                    sorted(
                        self.valid_ngram_keyword_pairs_list,
                        key=lambda x: x["support"],
                        reverse=True,
                    ),
                    f,
                    indent=4,
                )
            logger.info(
                "Saved valid ngram-keyword pairs (by support) to"
                f" {result_path}"
            )

            # save results to a json file, order by ngram itself
            result_path = os.path.join(
                result_folder, "valid_ngram_keyword_pairs_by_ngram.json"
            )
            with open(result_path, "w") as f:
                json.dump(
                    sorted(
                        self.valid_ngram_keyword_pairs_list,
                        key=lambda x: (x["ngram"]),
                    ),
                    f,
                    indent=4,
                )
            logger.info(
                f"Saved valid ngram-keyword pairs (by ngram) to {result_path}"
            )

            # save results to a json file, order by keyword
            result_path = os.path.join(
                result_folder, "valid_ngram_keyword_pairs_by_keyword.json"
            )
            with open(result_path, "w") as f:
                json.dump(
                    sorted(
                        self.valid_ngram_keyword_pairs_list,
                        key=lambda x: x["keyword"],
                    ),
                    f,
                    indent=4,
                )
            logger.info(
                "Saved valid ngram-keyword pairs (by keyword) to"
                f" {result_path}"
            )

        return self.valid_ngram_keyword_pairs_list


class InformationGainEvaluator(WordEvaluator):
    """
    Information Gain evaluator for discriminative pattern mining.

    Calculates how much knowing an n-gram reduces uncertainty about a keyword.
    """

    def __init__(self, min_score: float = 0.05, min_support: int = 30):
        super().__init__(min_score)
        self.min_support = min_support
        self.segment_ngrams = {}  # {segment_id: set of ngrams}
        self.segment_keywords = {}  # {segment_id: set of keywords}
        self.ngram_segments = defaultdict(set)  # {ngram: set of segment_ids}
        self.keyword_counts = {}
        self.total_segments = 0
        self.ngram_keyword_cooccurrence = defaultdict(lambda: defaultdict(int))
        self.keyword_entropies = {}

        self.valid_ngram_keyword_pairs_list = []

    def prepare_data(self, parse_result: ParseResult):
        """
        Prepare data structures for Information Gain calculation.

        Builds segment-level indices for binary presence of n-grams.

        :param parse_result: ParseResult instance from parser
        """
        logger.info("Building segment-level indices for Information Gain...")

        # This needs to be called with sentences passed separately
        # For now, we'll store the metadata
        self.keyword_counts = parse_result.keyword_counts
        self.total_segments = parse_result.total_segments

        logger.info(
            f"IG Evaluator prepared for {self.total_segments} segments"
        )

    def build_segment_index(self, sentences: Dict[str, "Sentence"]):
        """
        Build segment-level index from parsed sentences.

        :param sentences: Dictionary of {audio_name: Sentence}
        """
        logger.info("Building segment-level index for Information Gain...")

        segment_id = 0
        min_n, max_n = 1, 10  # Will be set from parser

        for sentence in sentences.values():
            for seg_idx, segment in enumerate(sentence.segments):
                # Track n-grams present in this segment (binary)
                ngrams_in_segment = set()

                # Extract all n-grams using sliding window
                for n in range(min_n, min(max_n + 1, len(segment) + 1)):
                    for i in range(len(segment) - n + 1):
                        ngram = tuple(segment[i : i + n])
                        ngrams_in_segment.add(ngram)
                        self.ngram_segments[ngram].add(segment_id)

                # Store segment data
                self.segment_ngrams[segment_id] = ngrams_in_segment
                self.segment_keywords[segment_id] = sentence.keywords

                # Update keyword counts
                for keyword in sentence.keywords:
                    self.keyword_counts[keyword] = (
                        self.keyword_counts.get(keyword, 0) + 1
                    )

                segment_id += 1

        self.total_segments = segment_id

        # Filter by minimum support
        initial_count = len(self.ngram_segments)
        filtered_ngrams = {
            ngram: segments
            for ngram, segments in self.ngram_segments.items()
            if len(segments) >= self.min_support
        }
        self.ngram_segments = filtered_ngrams

        logger.info(
            f"Total unique n-grams before filtering: {initial_count:,}"
        )
        logger.info(
            f"Total unique n-grams after min_support={self.min_support}:"
            f" {len(self.ngram_segments):,}"
        )

        # Precompute co-occurrences
        logger.info("Precomputing n-gram-keyword co-occurrences...")
        for ngram, seg_ids in self.ngram_segments.items():
            for seg_id in seg_ids:
                for keyword in self.segment_keywords[seg_id]:
                    self.ngram_keyword_cooccurrence[ngram][keyword] += 1

        # Precompute keyword entropies
        logger.info("Precomputing keyword entropies...")
        for keyword, count in self.keyword_counts.items():
            p_k = count / self.total_segments
            p_not_k = 1 - p_k
            if p_k > 0 and p_not_k > 0:
                self.keyword_entropies[keyword] = -p_k * np.log2(
                    p_k
                ) - p_not_k * np.log2(p_not_k)
            else:
                self.keyword_entropies[keyword] = 0.0

        logger.info(
            f"Precomputed data for {len(self.ngram_keyword_cooccurrence)}"
            " n-grams"
        )

    def calculate_score(self, ngram: Tuple[int, ...], keyword: str) -> float:
        """
        Calculate Information Gain: IG(P, K) = H(K) - H(K | P).

        :param ngram: N-gram tuple
        :param keyword: Keyword string
        :return: Information Gain score (0 to ~1)
        """
        if ngram not in self.ngram_segments:
            return 0.0

        if keyword not in self.keyword_counts:
            return 0.0

        N = self.total_segments
        if N == 0:
            return 0.0

        N_K = self.keyword_counts[keyword]
        segments_with_P = self.ngram_segments[ngram]
        N_P = len(segments_with_P)
        N_not_P = N - N_P

        if N_P == 0 or N_not_P == 0:
            return 0.0

        # Co-occurrence counts
        N_KP = self.ngram_keyword_cooccurrence[ngram].get(keyword, 0)
        N_K_not_P = N_K - N_KP

        # Global entropy H(K)
        H_K = self.keyword_entropies.get(keyword, 0.0)
        if H_K == 0.0:
            return 0.0

        # H(K | P present)
        P_K_given_P = N_KP / N_P if N_P > 0 else 0
        P_not_K_given_P = 1 - P_K_given_P

        if P_K_given_P == 0 or P_K_given_P == 1:
            H_K_given_P = 0.0
        else:
            H_K_given_P = -P_K_given_P * np.log2(
                P_K_given_P
            ) - P_not_K_given_P * np.log2(P_not_K_given_P)

        # H(K | P absent)
        P_K_given_not_P = N_K_not_P / N_not_P if N_not_P > 0 else 0
        P_not_K_given_not_P = 1 - P_K_given_not_P

        if P_K_given_not_P == 0 or P_K_given_not_P == 1:
            H_K_given_not_P = 0.0
        else:
            H_K_given_not_P = -P_K_given_not_P * np.log2(
                P_K_given_not_P
            ) - P_not_K_given_not_P * np.log2(P_not_K_given_not_P)

        # Conditional entropy H(K | P)
        P_P = N_P / N
        P_not_P = N_not_P / N
        H_K_given_P_overall = P_P * H_K_given_P + P_not_P * H_K_given_not_P

        # Information Gain
        IG = H_K - H_K_given_P_overall

        return max(0.0, IG)

    def find_top_ngrams_for_keyword(
        self, keyword: str, top_k: int = 10
    ) -> List[Tuple[Tuple[int, ...], float, int]]:
        """
        Find top n-grams for a keyword ranked by Information Gain.

        :param keyword: Target keyword
        :param top_k: Number of top results
        :return: [(ngram, ig_score, support), ...]
        """
        if keyword not in self.keyword_counts:
            return []

        results = []
        for ngram in self.ngram_segments:
            ig = self.calculate_score(ngram, keyword)
            if ig > self.min_score:
                support = len(self.ngram_segments[ngram])
                results.append((ngram, ig, support))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def find_valid_ngram_keyword_pairs(
        self,
        min_support: int = 50,
        min_ig: float = 0.05,
        result_folder: str = None,
    ) -> List[Dict]:
        """
        Find ngram-keyword pairs exceeding thresholds both for support and Information Gain.

        :param min_support: Minimum frequency of n-gram for consideration
        :param min_ig: Minimum Information Gain score for consideration
        :param result_folder: Optional folder path to save JSON results
        :return: List of dicts with keys: ngram, keyword, support, ig
        """
        if not self.ngram_segments:
            logger.warning(
                "IG Evaluator data not prepared. Call build_segment_index()"
                " first."
            )
            return []

        # Reset list to avoid accumulation across multiple calls
        self.valid_ngram_keyword_pairs_list = []

        for ngram in self.ngram_segments:
            if len(self.ngram_segments[ngram]) < min_support:
                continue

            # Check all keywords that co-occur with this ngram
            for keyword in self.ngram_keyword_cooccurrence[ngram]:
                ig = self.calculate_score(ngram, keyword)
                if ig >= min_ig:
                    self.valid_ngram_keyword_pairs_list.append({
                        "ngram": list(
                            ngram
                        ),  # Convert tuple to list for JSON serialization
                        "keyword": keyword,
                        "support": len(self.ngram_segments[ngram]),
                        "ig": float(ig),  # Ensure JSON serializable
                    })

        logger.info(
            f"Found {len(self.valid_ngram_keyword_pairs_list)} valid"
            " ngram-keyword pairs"
        )

        if result_folder:
            os.makedirs(result_folder, exist_ok=True)

            # save results to a json file, order by IG descending
            result_path = os.path.join(
                result_folder, "valid_ngram_keyword_pairs_by_ig.json"
            )
            with open(result_path, "w") as f:
                json.dump(
                    sorted(
                        self.valid_ngram_keyword_pairs_list,
                        key=lambda x: x["ig"],
                        reverse=True,
                    ),
                    f,
                    indent=4,
                )
            logger.info(
                f"Saved valid ngram-keyword pairs (by IG) to {result_path}"
            )

            # save results to a json file, order by support descending
            result_path = os.path.join(
                result_folder, "valid_ngram_keyword_pairs_by_support.json"
            )
            with open(result_path, "w") as f:
                json.dump(
                    sorted(
                        self.valid_ngram_keyword_pairs_list,
                        key=lambda x: x["support"],
                        reverse=True,
                    ),
                    f,
                    indent=4,
                )
            logger.info(
                "Saved valid ngram-keyword pairs (by support) to"
                f" {result_path}"
            )

            # save results to a json file, order by ngram length then ngram itself
            result_path = os.path.join(
                result_folder, "valid_ngram_keyword_pairs_by_ngram.json"
            )
            with open(result_path, "w") as f:
                json.dump(
                    sorted(
                        self.valid_ngram_keyword_pairs_list,
                        key=lambda x: (len(x["ngram"]), x["ngram"]),
                    ),
                    f,
                    indent=4,
                )
            logger.info(
                f"Saved valid ngram-keyword pairs (by ngram) to {result_path}"
            )

            # save results to a json file, order by keyword
            result_path = os.path.join(
                result_folder, "valid_ngram_keyword_pairs_by_keyword.json"
            )
            with open(result_path, "w") as f:
                json.dump(
                    sorted(
                        self.valid_ngram_keyword_pairs_list,
                        key=lambda x: x["keyword"],
                    ),
                    f,
                    indent=4,
                )
            logger.info(
                "Saved valid ngram-keyword pairs (by keyword) to"
                f" {result_path}"
            )

        return self.valid_ngram_keyword_pairs_list


# ==================== Main WordFinder Class ====================


class WordFinder:
    """
    Main class for word discovery using various methods.
    """

    DELIMITER = -1  # Frames without dog barks (segment delimiter)
    MIN_EXPORT_SUPPORT = 50  # Minimum segments for export
    MAX_SEGMENT_LENGTH_ENUMERATION = (
        20  # Max length for full segmentation enumeration
    )
    VIDEO_FILENAME_SUFFIX_LENGTH = (
        5  # Length of suffix to remove from video filenames: *_clip.mp4
    )

    # Boundary Model Parameters
    BOUNDARY_PRIOR = 0.3  # Prior probability for boundaries (boundaries are rarer than non-boundaries)
    SEGMENT_PENALTY = (
        -0.5
    )  # Penalty for creating segments (negative log-prob, encourages longer segments when positive)
    USE_LENGTH_NORMALIZATION = (
        True  # Normalize by segment length to avoid length bias
    )

    def __init__(
        self,
        transcription: Dict[str, List[int]],
        keywords: Dict[str, List[str]],
        ngram_range: Tuple[int, int] = (1, 10),
        min_support: int = 50,
        min_eval_score: float = 0.2,
        video_folders: list = [
            "/redgiant/data/bigdrive/_assets/dog_sentence_clips",
            "/redgiant/data/bigdrive/_assets/dog_sentence_clips2",
        ],
    ):
        self.transcription = transcription
        self.keywords = keywords
        self.sentences = {}  # {audio_name: Sentence}
        self.parser = None
        self.evaluator = None
        self.ngram_range = ngram_range
        self.min_support = min_support
        self.min_eval_score = min_eval_score

        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.parse_result = (
            None  # {"ngram_index":..., "keyword_counts":..., ...}
        )
        self.video_sentence_map = None

        sentence_video_map = self._build_video_mapping(video_folders)
        self._create_sentence_objects(
            transcription, keywords, sentence_video_map
        )
        self._video_path_sentence_map()
        self.get_statistics(print_stat=False)

    def get_statistics(self, print_stat: bool = True) -> None:
        """
        Print statistics about loaded sentences.

        :return: None
        """
        phone_durations = defaultdict(list)
        for sen in self.sentences.values():
            for phone, dur in zip(
                sen.transcription, sen.phone_duration_transcription
            ):
                phone_durations[phone].append(dur)
        self.phone_duration_stat = phone_durations

        if print_stat:
            print(f"Total sentences: {len(self.sentences)}")
            print("Phone Duration Statistics:")

            # Collect statistics for all phones
            phone_stats = {}
            for phone, durations in sorted(phone_durations.items()):
                durations_array = np.array(durations)
                avg_duration = np.mean(durations_array)
                median_duration = np.median(durations_array)
                std_duration = np.std(durations_array)
                min_duration = np.min(durations_array)
                max_duration = np.max(durations_array)
                p95_duration = np.percentile(durations_array, 95)
                count = len(durations)
                phone_stats[phone] = {
                    "mean": avg_duration,
                    "median": median_duration,
                    "std": std_duration,
                    "min": min_duration,
                    "max": max_duration,
                    "p95": p95_duration,
                    "count": count,
                }
                print(
                    f"Phone {phone}: count={count}, mean={avg_duration:.3f}s, "
                    f"median={median_duration:.3f}s, std={std_duration:.3f}s, "
                    f"min={min_duration:.3f}s, max={max_duration:.3f}s, "
                    f"95th={p95_duration:.3f}s"
                )

    def plot_phone_duration_dist(
        self, output_path: str = None, percentile_cap: float = 99.0
    ):
        """
        Plot violin plot for phone duration distributions.

        :param output_path: Path to save the plot (default: './phone_duration_violin.png')
        :param percentile_cap: Percentile cap for outliers (default: 99.0)
        """
        if (
            not hasattr(self, "phone_duration_stat")
            or not self.phone_duration_stat
        ):
            logger.warning(
                "No phone duration statistics available. Call get_statistics()"
                " first."
            )
            return

        # Prepare data
        phones = sorted(self.phone_duration_stat.keys())
        data = []
        labels = []

        for phone in phones:
            durations = self.phone_duration_stat[phone]
            # Cap at percentile to handle outliers
            cap_value = np.percentile(durations, percentile_cap)
            capped = [min(d, cap_value) for d in durations]
            data.append(capped)
            labels.append(f"Phone {phone}")

        # Create violin plot
        fig, ax = plt.subplots(figsize=(max(15, len(phones) * 0.8), 6))

        parts = ax.violinplot(data, showmedians=True, showextrema=True)

        # Customize violin plot colors
        for pc in parts["bodies"]:
            pc.set_facecolor("#8dd3c7")
            pc.set_alpha(0.7)

        ax.set_xticks(range(1, len(labels) + 1))
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_ylabel("Duration (seconds)")
        ax.set_xlabel("Phone ID")
        ax.set_title(
            f"Phone Duration Distributions (capped at {percentile_cap}th"
            " percentile)"
        )
        ax.grid(axis="y", alpha=0.3)

        plt.tight_layout()

        # Save plot
        if output_path is None:
            output_path = "./phone_duration_violin.png"

        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved phone duration violin plot to {output_path}")
        plt.close()

    def _build_video_mapping(self, video_folders: List[str]) -> Dict[str, str]:
        """
        Build mapping from sentence IDs to video file paths.

        :param video_folders: List of directories containing video files
        :return: Dictionary mapping sentence_id -> video_path
        """
        video_map = {}
        for folder in video_folders:
            if not os.path.exists(folder):
                logger.warning(f"Video folder not found: {folder}")
                continue

            for filename in os.listdir(folder):
                if not filename.endswith(".mp4"):
                    continue
                sentence_id = os.path.basename(filename).split(".")[0]
                # Remove suffix (last 5 characters)
                sentence_id = sentence_id[: -self.VIDEO_FILENAME_SUFFIX_LENGTH]
                video_map[sentence_id] = os.path.join(folder, filename)

        logger.info(
            f"Loaded {len(video_map)} video files from {len(video_folders)}"
            " folders"
        )
        return video_map

    def _create_sentence_objects(
        self, transcription: dict, keywords: dict, sentence_video_map: dict
    ):
        """
        Create Sentence objects from transcription and keywords.

        :return: None
        """
        for audio_name in transcription:
            if audio_name not in keywords:
                continue

            if isinstance(transcription[audio_name], dict):
                raw_trans = transcription[audio_name]["transcription"]
                duration = transcription[audio_name].get("duration", 0.0)
                audio_path = transcription[audio_name].get("audio_path", "")
            elif isinstance(transcription[audio_name], list):
                raw_trans = transcription[audio_name]
                duration = 0.0
                audio_path = ""
            else:
                logger.warning(
                    f"Unknown transcription format for {audio_name}"
                )
                continue

            if len(raw_trans) == 0:
                continue

            video_path = sentence_video_map.get(audio_name, None)

            sentence = Sentence(
                audio_name=audio_name,
                raw_transcription=raw_trans,
                audio_path=audio_path,
                duration=duration,
                keywords=set(keywords[audio_name]),
                video_path=video_path,
                smooth=None,
            )

            self.sentences[audio_name] = sentence

    def _video_path_sentence_map(self) -> Dict[str, Sentence]:
        """
        Create mapping from video file paths to Sentence objects.

        :return: Dictionary mapping video_path -> Sentence
        """
        video_sentence_map = {}
        for sentence in self.sentences.values():
            if sentence.video_path:
                video_sentence_map[sentence.video_path] = sentence
        self.video_sentence_map = video_sentence_map

    def set_min_support(self, min_support: int):
        self.min_support = min_support
        if self.evaluator:
            self.evaluator.min_support = min_support

    def set_min_eval_score(self, min_eval_score: float):
        self.min_eval_score = min_eval_score
        if self.evaluator:
            self.evaluator.min_score = min_eval_score

    def set_parser(self, method: str, **kwargs):
        if method == "slide_window":
            self.parser = SlidingWindowParser(**kwargs)
        elif method == "boundary_prob":
            self.parser = BoundaryModelParser(**kwargs)
        else:
            raise ValueError(f"Unknown parser method: {method}")

    def set_evaluator(self, method: str):
        if method == "npmi":
            self.evaluator = NPMIEvaluator(
                min_score=self.min_eval_score, min_support=self.min_support
            )
        elif method == "ig":
            self.evaluator = InformationGainEvaluator(
                min_score=self.min_eval_score, min_support=self.min_support
            )
        else:
            raise ValueError(f"Unknown evaluator method: {method}")

    def parse(self) -> ParseResult:
        if not self.parser:
            raise ValueError("Parser not set. Call set_parser() first.")
        self.parse_result = self.parser.parse_sentences(self.sentences)
        return self.parse_result

    def get_words(self, result_folder: str = None):
        if not self.evaluator:
            raise ValueError("Evaluator not set. Call set_evaluator() first.")
        if not self.parse_result:
            raise ValueError("No parse result available. Call parse() first.")

        self.evaluator.prepare_data(self.parse_result)
        if result_folder:
            result_folder = os.path.join(
                result_folder, f"word_finding_results_{self.timestamp}"
            )
            self.evaluator.find_valid_ngram_keyword_pairs(
                result_folder=result_folder
            )
        else:
            return self.evaluator.find_valid_ngram_keyword_pairs()

    def generate_found_words_video_samples(
        self, result_folder: str, num_samples: int = 5
    ):
        if not self.parse_result:
            raise ValueError("No parse result available. Call parse() first.")
        if result_folder:
            result_folder = os.path.join(
                result_folder, f"word_finding_results_{self.timestamp}"
            )
        else:
            raise ValueError(
                "result_folder must be specified to save video samples."
            )

        os.makedirs(result_folder, exist_ok=True)

        for (
            ngram,
            keyword_video_paths,
        ) in self.parse_result.ngram_keyword_video_paths.items():
            for keyword, video_paths in keyword_video_paths.items():
                ngram_str = "_".join(map(str, ngram))
                keyword_str = keyword.replace(" ", "_")
                ngram_keyword_folder = os.path.join(
                    result_folder, f"{ngram_str}_{keyword_str}"
                )
                os.makedirs(ngram_keyword_folder, exist_ok=True)
                sampled_paths = random.sample(
                    video_paths, min(num_samples, len(video_paths))
                )
                for video_path in sampled_paths:
                    video_filename = os.path.basename(video_path)
                    dest_path = os.path.join(
                        ngram_keyword_folder, video_filename
                    )
                    shutil.copy(video_path, dest_path)
                logger.info(
                    f"Copied {len(sampled_paths)} video samples for ngram"
                    f" {ngram_str} and keyword '{keyword}' to"
                    f" {ngram_keyword_folder}"
                )
        return

    def save_parse_results(self, filepath: str):
        if not self.parse_result:
            raise ValueError("No parse result available. Call parse() first.")
        self.parse_result.save(filepath)

    def load_parse_results(self, filepath: str):
        self.parse_result = ParseResult.load(filepath)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Canine word discovery via EM boundary model + NPMI")
    parser.add_argument("--transcription", required=True, help="Path to transcription JSON {clip_id: [int, ...]}")
    parser.add_argument("--keywords",      required=True, help="Path to keywords JSON {clip_id: [str, ...]}")
    parser.add_argument("--output",        default="./output", help="Output directory")
    parser.add_argument("--ngram_max",     type=int,   default=10)
    parser.add_argument("--em_iters",      type=int,   default=10)
    parser.add_argument("--min_support",   type=int,   default=50)
    parser.add_argument("--min_npmi",      type=float, default=0.18)
    args = parser.parse_args()

    with open(args.transcription) as f:
        transcription = json.load(f)
    with open(args.keywords) as f:
        keywords = json.load(f)
    transcription = {k: v for k, v in transcription.items() if k in keywords}

    word_finder = WordFinder(transcription, keywords)
    word_finder.set_parser(
        "boundary_prob",
        ngram_range=(1, args.ngram_max),
        max_iterations=args.em_iters,
        use_keyword_context=True,
    )
    word_finder.set_min_support(args.min_support)
    word_finder.set_min_eval_score(args.min_npmi)
    word_finder.set_evaluator("npmi")
    word_finder.parse()
    word_finder.get_words(result_folder=args.output)
    word_finder.save_parse_results(
        filepath=f"{args.output}/word_finding_results_{word_finder.timestamp}/parse_results.json"
    )


if __name__ == "__main__":
    main()
