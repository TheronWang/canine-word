import json
import os
import random
import time
from copy import deepcopy
from pathlib import Path
import pandas as pd
import pytorch_lightning as pl
import scipy
import torch
import torch.nn as nn
from torchaudio.transforms import AmplitudeToDB, MelSpectrogram

from desed_task.data_augm import mixup
from desed_task.utils.scaler import TorchScaler
import numpy as np
import torchmetrics

from .utils import (
    batched_decode_preds,
    log_sedeval_metrics,
)
from desed_task.evaluation.evaluation_measures import (
    compute_per_intersection_macro_f1,
    compute_psds_from_operating_points,
    compute_psds_from_scores
)

import sed_scores_eval

from desed_task.nnet.CRNN_e2e import CRNN
from desed_task.utils.encoder import ManyHotEncoder
from desed_task.dataio.datasets_atst_sed import SEDTransform, ATSTTransform, read_audio
from desed_task.utils.scaler import TorchScaler
from .evaluate2 import get_seds_pred, get_other_metrics


# ------------------------------------ inference

class BEATSSEDFeatureExtractor(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.sed_feat_extractor = SEDTransform(config["feats"])
        self.scaler = TorchScaler(
            "instance",
            config["scaler"]["normtype"],
            config["scaler"]["dims"],
        )
        self.sed_feat_extractor.transform.to('cuda:0')

    def take_log(self, mels):
        amp_to_db = AmplitudeToDB(stype="amplitude")
        amp_to_db.amin = 1e-5  # amin= 1e-5 as in librosa
        return amp_to_db(mels).clamp(min=-50, max=80)  # clamp to reproduce old code

    def forward(self, mixture):
        mixture = mixture.unsqueeze(0)  # fake batch size
        sed_feats = self.sed_feat_extractor(mixture)
        sed_feats = self.scaler(self.take_log(sed_feats))
        return sed_feats, mixture


class BEATSSEDInferencer(nn.Module):
    """Inference module for ATST-SED
    """

    def __init__(
            self,
            config,
            encoder,
            overlap_dur=5,
    ):
        super().__init__()

        self.config = config
        # Initialize label encoder
        self.label_encoder = encoder
        # Initialize feature extractor
        self.feature_extractor = BEATSSEDFeatureExtractor(config)

        # Initial parameters
        self.audio_dur = 10  # this value is fixed because ATST-SED is trained on 10-second audio, if you want to change it, you need to retrain the model
        self.overlap_dur = overlap_dur
        self.fs = config["data"]["fs"]

        # Unfolder for splitting audio into chunks
        self.unfolder = nn.Unfold(kernel_size=(self.fs * self.audio_dur, 1), stride=(self.fs * self.overlap_dur, 1))

        # self.hard_threshold = [hard_threshold] * len(self.label_encoder.labels) if not isinstance(hard_threshold, list) else hard_threshold

        self.hard_threshold = None

    def get_logmel(self, wav_file):
        mixture, onset_s, offset_s, padded_indx = read_audio(
            wav_file, False, False, None
        )
        sed_feats, _ = self.feature_extractor(mixture)
        return sed_feats[0].detach().cpu().numpy()

    def forward(self, mixture, model):

        # split wav into chunks with overlap
        if mixture.shape[0] <= (self.audio_dur * self.fs):
            inference_chunks = [mixture]
            padding_frames = 0
            mixture_pad = mixture.clone()
        else:
            # pad the mixtures
            mixture = mixture.unsqueeze(0).unsqueeze(0).unsqueeze(-1)
            total_chunks = (mixture.numel() - ((self.audio_dur - self.overlap_dur) * self.fs)) // (
                    self.overlap_dur * self.fs) + 1
            total_length = total_chunks * self.overlap_dur * self.fs + (self.audio_dur - self.overlap_dur) * self.fs
            mixture_pad = torch.nn.functional.pad(mixture, (0, 0, 0, total_length - mixture.numel()))
            padding_frames = self.time2frame(total_length - mixture.numel())
            inference_chunks = self.unfolder(mixture_pad)
            inference_chunks = inference_chunks.squeeze(0).T

            if (total_length - mixture.numel()) > 0:
                padding_frames = max(padding_frames, 1)

        # inference result for each chunk
        sed_results = []
        for chunk in inference_chunks:
            sed_feats, mixture = self.feature_extractor(chunk)
            # sed_feats = sed_feats.transpose(1, 2)
            chunk_result, _ = model(sed_feats, mixture)
            sed_results.append(chunk_result)

        if self.hard_threshold is None:
            # return sed_results  # If no threshold given, return soft results and you can customize the threshold later
            chunk_scores = [self.get_post_scores(r.detach().cpu()) for r in sed_results]
            return self.unify_scores(chunk_scores=chunk_scores,
                                     total_frames=self.time2frame(mixture_pad.numel()),
                                     padding_frames=padding_frames)
        else:
            assert False

    def get_post_scores(self, strong_preds):
        strong_preds = strong_preds[0]  # only support single input (bsz=1)
        # smoothed_preds = []
        # for score, fil_val in zip(strong_preds, self.config["training"]["median_window"]):
        #     score = scipy.ndimage.filters.median_filter(score[:, np.newaxis], (fil_val, 1))
        #     smoothed_preds.append(score)
        # smoothed_preds = np.concatenate(smoothed_preds, axis=1)
        # return smoothed_preds.T

        return strong_preds.numpy()

    def unify_scores(self, chunk_scores, total_frames, padding_frames):
        # TODO: 只能用在一个类
        C, T = chunk_scores[0].shape
        if len(chunk_scores) == 1:
            return chunk_scores[0][0]
        else:
            decisions = np.zeros((C, total_frames))
            hop_frame = self.time2frame(self.overlap_dur * self.fs)
            decisions[:, 0 * hop_frame: 0 * hop_frame + T] += chunk_scores[0]
            # for i in range(1, len(chunk_scores)):
            #     decisions[:, i * hop_frame: i * hop_frame + T] += chunk_scores[i]
            #     decisions[:, i * hop_frame: i * hop_frame + T - hop_frame] *= 0.5

            for i in range(1, len(chunk_scores)):
                assert decisions[:, i * hop_frame: i * hop_frame + T].shape == chunk_scores[i].shape
                decisions[:, i * hop_frame: i * hop_frame + T] = np.maximum(chunk_scores[i], decisions[:,
                                                                                             i * hop_frame: i * hop_frame + T])
            return decisions[0, : -padding_frames]

    def time2frame(self, time):
        return int(int((time / self.label_encoder.frame_hop)) / self.label_encoder.net_pooling)



# --------------------------------------- inference



class SEDTask4(pl.LightningModule):
    def __init__(
            self,
            hparams,
            encoder,
            sed_student,
            opt=None,
            train_data=None,
            valid_data=None,
            test_data=None,
            train_sampler=None,
            scheduler=None,
            fast_dev_run=False,
            evaluation=False,
            sed_teacher=None,

            seds_duration_dict=None,
            seds_ground_truth_dict=None,
            valid_results_save_path=None,
            pin_m=False
    ):
        super(SEDTask4, self).__init__()

        self.seds_duration_dict = seds_duration_dict
        self.seds_ground_truth_dict = seds_ground_truth_dict
        self.valid_results_save_path = valid_results_save_path
        self.pin_m = pin_m

        self.hparams.update(hparams)

        self.encoder = encoder
        self.sed_student = sed_student
        self.sed_teacher = deepcopy(sed_student) if sed_teacher is None else sed_teacher

        self.opt = opt
        self.train_data = train_data
        self.valid_data = valid_data
        self.test_data = test_data
        self.train_sampler = train_sampler
        self.scheduler = scheduler
        self.fast_dev_run = fast_dev_run
        self.evaluation = evaluation

        if self.fast_dev_run:
            self.num_workers = 1
        else:
            self.num_workers = self.hparams["training"]["num_workers"]

        feat_params = self.hparams["feats"]
        self.mel_spec = MelSpectrogram(
            sample_rate=feat_params["sample_rate"],
            n_fft=feat_params["n_window"],
            win_length=feat_params["n_window"],
            hop_length=feat_params["hop_length"],
            f_min=feat_params["f_min"],
            f_max=feat_params["f_max"],
            n_mels=feat_params["n_mels"],
            window_fn=torch.hamming_window,
            wkwargs={"periodic": False},
            power=1,
        )
        for param in self.sed_teacher.parameters():
            param.detach_()

        # instantiating losses
        self.supervised_loss = torch.nn.BCELoss()
        if hparams["training"]["self_sup_loss"] == "mse":
            self.selfsup_loss = torch.nn.MSELoss()
        elif hparams["training"]["self_sup_loss"] == "bce":
            self.selfsup_loss = torch.nn.BCELoss()
        else:
            raise NotImplementedError

        # 注释以下两个计算weak的部分
        # # for weak labels we simply compute f1 score
        # self.get_weak_student_f1_seg_macro = torchmetrics.classification.f_beta.MultilabelF1Score(
        #     len(self.encoder.labels),
        #     average="macro",
        #     compute_on_step=False,
        # )
        #
        # self.get_weak_teacher_f1_seg_macro = torchmetrics.classification.f_beta.MultilabelF1Score(
        #     len(self.encoder.labels),
        #     average="macro",
        #     compute_on_step=False,
        # )

        self.scaler = self._init_scaler()
        # buffer for event based scores which we compute using sed-eval

        self.val_thds = [0.5]
        self.val_scores_postprocessed_buffer_student_synth = {}
        self.val_scores_postprocessed_buffer_teacher_synth = {}
        self.val_scores_postprocessed_buffer_student_real = {}
        self.val_scores_postprocessed_buffer_teacher_real = {}

        self.val_loss_weak_student = []
        self.val_loss_weak_teacher = []
        self.val_loss_synth_student = []
        self.val_loss_synth_teacher = []
        self.val_loss_real_student = []
        self.val_loss_real_teacher = []

        self.val_buffer_student_synth = {
            k: pd.DataFrame() for k in self.hparams["training"]["val_thresholds"]
        }

        test_n_thresholds = self.hparams["training"]["n_test_thresholds"]
        test_thresholds = np.arange(
            1 / (test_n_thresholds * 2), 1, 1 / test_n_thresholds
        )
        self.test_psds_buffer_student = {k: pd.DataFrame() for k in test_thresholds}
        self.test_psds_buffer_teacher = {k: pd.DataFrame() for k in test_thresholds}
        self.decoded_student_05_buffer = pd.DataFrame()
        self.decoded_teacher_05_buffer = pd.DataFrame()
        self.test_scores_raw_buffer_student = {}
        self.test_scores_raw_buffer_teacher = {}
        self.test_scores_postprocessed_buffer_student = {}
        self.test_scores_postprocessed_buffer_teacher = {}

        self.inference_model = BEATSSEDInferencer(config=hparams, encoder=encoder)
        self.valid_teacher_results = {}
        self.valid_student_results = {}
        self.frames_per_second = self.hparams["data"]["fs"] / (
                    self.hparams["data"]["net_subsample"] * self.hparams["feats"]["hop_length"])
        self.my_step_count = 0
        self.valid_teacher_loss = []
        self.valid_student_loss = []

    _exp_dir = None

    @property
    def exp_dir(self):
        if self._exp_dir is None:
            try:
                self._exp_dir = self.logger.log_dir
            except Exception as e:
                self._exp_dir = self.hparams["log_dir"]
        return self._exp_dir

    def lr_scheduler_step(self, scheduler, optimizer_idx, metric):
        scheduler.step()

    def update_ema(self, alpha, global_step, model, ema_model):
        # Use the true average until the exponential average is more correct
        alpha = min(1 - 1 / (global_step + 1), alpha)
        for (k, ema_params), params in zip(ema_model.named_parameters(), model.parameters()):
            ema_params.data.mul_(alpha).add_(params.data, alpha=1 - alpha)

    def _init_scaler(self):
        if self.hparams["scaler"]["statistic"] == "instance":
            scaler = TorchScaler(
                "instance",
                self.hparams["scaler"]["normtype"],
                self.hparams["scaler"]["dims"],
            )

            return scaler
        elif self.hparams["scaler"]["statistic"] == "dataset":
            # we fit the scaler
            scaler = TorchScaler(
                "dataset",
                self.hparams["scaler"]["normtype"],
                self.hparams["scaler"]["dims"],
            )
        else:
            raise NotImplementedError
        if self.hparams["scaler"]["savepath"] is not None:
            if os.path.exists(self.hparams["scaler"]["savepath"]):
                scaler = torch.load(self.hparams["scaler"]["savepath"])
                print(
                    "Loaded Scaler from previous checkpoint from {}".format(
                        self.hparams["scaler"]["savepath"]
                    )
                )
                return scaler

        self.train_loader = self.train_dataloader()
        scaler.fit(
            self.train_loader,
            transform_func=lambda x: self.take_log(self.mel_spec(x[0])),
        )

        if self.hparams["scaler"]["savepath"] is not None:
            torch.save(scaler, self.hparams["scaler"]["savepath"])
            print(
                "Saving Scaler from previous checkpoint at {}".format(
                    self.hparams["scaler"]["savepath"]
                )
            )
            return scaler

    def take_log(self, mels):
        amp_to_db = AmplitudeToDB(stype="amplitude")
        amp_to_db.amin = 1e-5  # amin= 1e-5 as in librosa
        return amp_to_db(mels).clamp(min=-50, max=80)  # clamp to reproduce old code

    def detect(self, mel_feats, audio, model):
        return model(self.scaler(self.take_log(mel_feats)), audio)

    def training_step(self, batch, batch_indx):
        if self.pin_m: torch.cuda.empty_cache()
        audio, labels, padded_indxs, _ = batch

        # p = torch.cuda.get_device_properties(0)
        # info = f'000CUDA:{0} ({p.name}, {p.total_memory / (1 << 20):.0f}MiB)\n'
        # print(info)

        if len(self.hparams["training"]["batch_size"]) == 4:
            indx_strong, indx_synth, indx_weak, indx_unlabelled = self.hparams["training"]["batch_size"]
            indx_synth = indx_strong + indx_synth
        else:
            indx_synth, indx_weak, indx_unlabelled = self.hparams["training"]["batch_size"]

        sed_feats = self.mel_spec(audio)
        batch_num = sed_feats.shape[0]
        # deriving masks for each dataset
        strong_mask = torch.zeros(batch_num).to(sed_feats).bool()
        weak_mask = torch.zeros(batch_num).to(sed_feats).bool()
        unlabelled_mask = torch.zeros(batch_num).to(sed_feats).bool()
        strong_mask[:indx_synth] = 1
        weak_mask[indx_synth: indx_weak + indx_synth] = 1
        unlabelled_mask[indx_weak + indx_synth:] = 1

        # deriving weak labels
        labels_weak = (torch.sum(labels[weak_mask], -1) > 0).float()

        # 1-batch augmentation
        mixup_type = self.hparams["training"].get("mixup")
        if mixup_type is not None and 0.5 > random.random():
            sed_feats[weak_mask], labels_weak = mixup(
                sed_feats[weak_mask], labels_weak, mixup_label_type="soft"
            )
            sed_feats[strong_mask], labels[strong_mask] = mixup(
                sed_feats[strong_mask], labels[strong_mask], mixup_label_type="soft"
            )
            sed_feats[unlabelled_mask], _, = mixup(
                sed_feats[unlabelled_mask], labels[unlabelled_mask], mixup_label_type="soft"
            )

        # p = torch.cuda.get_device_properties(0)
        # info = f'111CUDA:{0} ({p.name}, {p.total_memory / (1 << 20):.0f}MiB)\n'
        # print(info)

        # sed student forward
        strong_preds_student, weak_preds_student = self.detect(
            sed_feats, audio, self.sed_student,
        )

        # p = torch.cuda.get_device_properties(0)
        # info = f'222CUDA:{0} ({p.name}, {p.total_memory / (1 << 20):.0f}MiB)\n'
        # print(info)
        # exit()

        # supervised loss on strong labels
        loss_strong = self.supervised_loss(strong_preds_student[strong_mask], labels[strong_mask])

        # supervised loss on weakly labelled
        loss_weak = self.supervised_loss(weak_preds_student[weak_mask], labels_weak)
        # total supervised loss
        tot_loss_supervised = loss_strong + loss_weak

        with torch.no_grad():
            strong_preds_teacher, weak_preds_teacher = self.detect(
                sed_feats, audio, self.sed_teacher
            )
            loss_strong_teacher = self.supervised_loss(
                strong_preds_teacher[strong_mask], labels[strong_mask]
            )

            loss_weak_teacher = self.supervised_loss(
                weak_preds_teacher[weak_mask], labels_weak
            )

        warmup = self.scheduler[0]["scheduler"]._get_scaling_factor()
        weight = (
                self.hparams["training"]["const_max"]
                * warmup
        )

        strong_self_sup_loss = self.selfsup_loss(
            strong_preds_student, strong_preds_teacher.detach()
        )
        weak_self_sup_loss = self.selfsup_loss(
            weak_preds_student, weak_preds_teacher.detach()
        )

        tot_self_loss = (strong_self_sup_loss + weak_self_sup_loss) * weight

        # we apply consistency between the predictions, use the scheduler for learning rate (to be changed ?)

        step_num = self.scheduler[0]["scheduler"].step_num
        cnn_lr = self.opt[-1].param_groups[0]["lr"]
        tfm_lr = self.opt[-1].param_groups[-1]["lr"]

        tot_loss = tot_loss_supervised + tot_self_loss

        self.log("train/student/loss_strong", loss_strong)
        self.log("train/student/loss_weak", loss_weak)
        self.log("train/teacher/loss_strong", loss_strong_teacher)
        self.log("train/teacher/loss_weak", loss_weak_teacher)
        self.log("train/step", step_num, prog_bar=True)
        self.log("train/student/weak_self_sup_loss", weak_self_sup_loss)
        self.log("train/student/strong_self_sup_loss", strong_self_sup_loss)
        self.log("train/student/tot_self_loss", tot_self_loss, prog_bar=True)
        self.log("train/weight", weight)
        self.log("train/student/tot_supervised", tot_loss_supervised, prog_bar=True)
        self.log("train/tfm_lr", tfm_lr)
        self.log("train/cnn_lr", cnn_lr, prog_bar=True)

        self.my_step_count = step_num
        return tot_loss

    def on_before_zero_grad(self, *args, **kwargs):
        # update EMA teacher
        self.update_ema(
            self.hparams["training"]["ema_factor"],
            self.scheduler[0]["scheduler"].step_num,
            self.sed_student,
            self.sed_teacher,
        )

    def valid_frame_to_time(self, frame):
        frame = frame * self.net_pooling / (self.fs / self.frame_hop)
        return np.clip(frame, a_min=0, a_max=self.audio_len)

    def validation_step(self, batch, batch_indx):
        name = batch['name'][0]
        mixture = batch['mixture'].squeeze(0)
        mixture.to('cuda:0')
        target = batch['target'].squeeze(0)
        target.to('cuda:0')
        target = target.to(torch.float32)

        teacher_sed_results = self.inference_model(mixture=mixture, model=self.sed_teacher)
        student_sed_results = self.inference_model(mixture=mixture, model=self.sed_student)
        assert len(teacher_sed_results.shape) == 1
        assert len(student_sed_results.shape) == 1
        teacher_count = min(teacher_sed_results.shape[0], target.shape[0])
        student_count = min(student_sed_results.shape[0], target.shape[0])
        teacher_sed_results = teacher_sed_results[0: teacher_count]
        student_sed_results = student_sed_results[0: student_count]
        self.valid_teacher_results[name] = teacher_sed_results
        self.valid_student_results[name] = student_sed_results

        pred = torch.tensor(teacher_sed_results, device='cuda:0', dtype=torch.float32)
        loss_strong_teacher = self.supervised_loss(pred, target)
        pred = torch.tensor(student_sed_results, device='cuda:0', dtype=torch.float32)
        loss_strong_student = self.supervised_loss(pred, target)
        self.valid_teacher_loss.append(loss_strong_teacher.item())
        self.valid_student_loss.append(loss_strong_student.item())

    def validation_epoch_end(self, outputs):
        def get_result(test_results, seds_ground_truth0, seds_durations0):
            seds_result = get_seds_pred(test_results=test_results,
                                        frames_per_second=self.frames_per_second,
                                        seds_ground_truth0=seds_ground_truth0,
                                        seds_durations0=seds_durations0)
            work_thrs = np.arange(0.05, 1., 0.05).tolist()
            for key1 in seds_result.keys():
                for key2 in seds_result[key1].keys():
                    if key2.find('threshold') < 0: continue
                    if isinstance(seds_result[key1][key2], list):
                        work_thrs.extend(seds_result[key1][key2])
                    else:
                        work_thrs.append(seds_result[key1][key2])

            work_thrs = list(set(work_thrs))
            work_thrs.sort()
            other_result = []
            for thr in work_thrs:
                other_result.append(get_other_metrics(thr=thr,
                                                      pred=test_results,
                                                      ground_truth=seds_ground_truth0,
                                                      valid_durations=seds_durations0,
                                                      frames_per_second=self.frames_per_second))
            return seds_result, other_result

        teacher_seds_result = {}
        teacher_other_result = {}
        student_seds_result = {}
        student_other_result = {}

        for key in self.seds_duration_dict.keys():
            cur_valid_teacher_results = {k: v for k, v in self.valid_teacher_results.items() if k in self.seds_duration_dict[key].keys()}
            cur_valid_student_results = {k: v for k, v in self.valid_student_results.items() if k in self.seds_duration_dict[key].keys()}
            cur_teacher_seds_result, cur_teacher_other_result = get_result(cur_valid_teacher_results,
                                                                           seds_ground_truth0=self.seds_ground_truth_dict[key],
                                                                           seds_durations0=self.seds_duration_dict[key])
            cur_student_seds_result, cur_student_other_result = get_result(cur_valid_student_results,
                                                                           seds_ground_truth0=self.seds_ground_truth_dict[key],
                                                                           seds_durations0=self.seds_duration_dict[key])
            teacher_seds_result[key] = cur_teacher_seds_result
            teacher_other_result[key] = cur_teacher_other_result
            student_seds_result[key] = cur_student_seds_result
            student_other_result[key] = cur_student_other_result

        cur_result = {
            'epoch': self.current_epoch,
            'step': self.my_step_count,
            'teacher_loss': sum(self.valid_teacher_loss),
            'student_loss': sum(self.valid_student_loss),
            'teacher_seds_result': teacher_seds_result,
            'teacher_other_result': teacher_other_result,
            'student_seds_result': student_seds_result,
            'student_other_result': student_other_result
        }
        with open(self.valid_results_save_path, 'a+', encoding='utf-8') as file:
            file.write(json.dumps(cur_result, ensure_ascii=False))
            file.write('\n')
            file.flush()

        self.valid_teacher_results = {}
        self.valid_student_results = {}
        self.valid_teacher_loss = []
        self.valid_student_loss = []

    def on_save_checkpoint(self, checkpoint):
        checkpoint["sed_student"] = self.sed_student.state_dict()
        checkpoint["sed_teacher"] = self.sed_teacher.state_dict()
        return checkpoint

    def test_step(self, batch, batch_indx):
        pass

    def on_test_epoch_end(self):
        pass

    def on_test_epoch_end0(self):
        # pub eval dataset
        save_dir = os.path.join(self.exp_dir, "metrics_test")

        if self.evaluation:
            # only save prediction scores
            save_dir_student_raw = os.path.join(save_dir, "student_scores", "raw")
            sed_scores_eval.io.write_sed_scores(self.test_scores_raw_buffer_student, save_dir_student_raw)
            print(f"\nRaw scores for student saved in: {save_dir_student_raw}")

            save_dir_student_postprocessed = os.path.join(save_dir, "student_scores", "postprocessed")
            sed_scores_eval.io.write_sed_scores(self.test_scores_postprocessed_buffer_student,
                                                save_dir_student_postprocessed)
            print(f"\nPostprocessed scores for student saved in: {save_dir_student_postprocessed}")

            save_dir_teacher_raw = os.path.join(save_dir, "teacher_scores", "raw")
            sed_scores_eval.io.write_sed_scores(self.test_scores_raw_buffer_teacher, save_dir_teacher_raw)
            print(f"\nRaw scores for teacher saved in: {save_dir_teacher_raw}")

            save_dir_teacher_postprocessed = os.path.join(save_dir, "teacher_scores", "postprocessed")
            sed_scores_eval.io.write_sed_scores(self.test_scores_postprocessed_buffer_teacher,
                                                save_dir_teacher_postprocessed)
            print(f"\nPostprocessed scores for teacher saved in: {save_dir_teacher_postprocessed}")

        else:
            # calculate the metrics
            ground_truth = sed_scores_eval.io.read_ground_truth_events(self.hparams["data"]["test_tsv"])
            audio_durations = sed_scores_eval.io.read_audio_durations(self.hparams["data"]["test_dur"])
            if self.fast_dev_run:
                ground_truth = {
                    audio_id: ground_truth[audio_id]
                    for audio_id in self.test_scores_postprocessed_buffer_student
                }
                audio_durations = {
                    audio_id: audio_durations[audio_id]
                    for audio_id in self.test_scores_postprocessed_buffer_student
                }
            else:
                # drop audios without events
                ground_truth = {
                    audio_id: gt for audio_id, gt in ground_truth.items()
                    if len(gt) > 0
                }
                audio_durations = {
                    audio_id: audio_durations[audio_id]
                    for audio_id in ground_truth.keys()
                }
            psds1_student_psds_eval = compute_psds_from_operating_points(
                self.test_psds_buffer_student,
                self.hparams["data"]["test_tsv"],
                self.hparams["data"]["test_dur"],
                dtc_threshold=0.7,
                gtc_threshold=0.7,
                alpha_ct=0,
                alpha_st=1,
                save_dir=os.path.join(save_dir, "student", "scenario1"),
            )
            psds1_student_sed_scores_eval = compute_psds_from_scores(
                self.test_scores_postprocessed_buffer_student,
                ground_truth,
                audio_durations,
                dtc_threshold=0.7,
                gtc_threshold=0.7,
                cttc_threshold=None,
                alpha_ct=0,
                alpha_st=1,
                save_dir=os.path.join(save_dir, "student", "scenario1"),
            )

            psds2_student_psds_eval = compute_psds_from_operating_points(
                self.test_psds_buffer_student,
                self.hparams["data"]["test_tsv"],
                self.hparams["data"]["test_dur"],
                dtc_threshold=0.1,
                gtc_threshold=0.1,
                cttc_threshold=0.3,
                alpha_ct=0.5,
                alpha_st=1,
                save_dir=os.path.join(save_dir, "student", "scenario2"),
            )
            psds2_student_sed_scores_eval = compute_psds_from_scores(
                self.test_scores_postprocessed_buffer_student,
                ground_truth,
                audio_durations,
                dtc_threshold=0.1,
                gtc_threshold=0.1,
                cttc_threshold=0.3,
                alpha_ct=0.5,
                alpha_st=1,
                save_dir=os.path.join(save_dir, "student", "scenario2"),
            )

            psds1_teacher_psds_eval = compute_psds_from_operating_points(
                self.test_psds_buffer_teacher,
                self.hparams["data"]["test_tsv"],
                self.hparams["data"]["test_dur"],
                dtc_threshold=0.7,
                gtc_threshold=0.7,
                alpha_ct=0,
                alpha_st=1,
                save_dir=os.path.join(save_dir, "teacher", "scenario1"),
            )
            psds1_teacher_sed_scores_eval = compute_psds_from_scores(
                self.test_scores_postprocessed_buffer_teacher,
                ground_truth,
                audio_durations,
                dtc_threshold=0.7,
                gtc_threshold=0.7,
                cttc_threshold=None,
                alpha_ct=0,
                alpha_st=1,
                save_dir=os.path.join(save_dir, "teacher", "scenario1"),
            )

            psds2_teacher_psds_eval = compute_psds_from_operating_points(
                self.test_psds_buffer_teacher,
                self.hparams["data"]["test_tsv"],
                self.hparams["data"]["test_dur"],
                dtc_threshold=0.1,
                gtc_threshold=0.1,
                cttc_threshold=0.3,
                alpha_ct=0.5,
                alpha_st=1,
                save_dir=os.path.join(save_dir, "teacher", "scenario2"),
            )
            psds2_teacher_sed_scores_eval = compute_psds_from_scores(
                self.test_scores_postprocessed_buffer_teacher,
                ground_truth,
                audio_durations,
                dtc_threshold=0.1,
                gtc_threshold=0.1,
                cttc_threshold=0.3,
                alpha_ct=0.5,
                alpha_st=1,
                save_dir=os.path.join(save_dir, "teacher", "scenario2"),
            )

            event_macro_student = log_sedeval_metrics(
                self.decoded_student_05_buffer,
                self.hparams["data"]["test_tsv"],
                os.path.join(save_dir, "student"),
            )[0]

            event_macro_teacher = log_sedeval_metrics(
                self.decoded_teacher_05_buffer,
                self.hparams["data"]["test_tsv"],
                os.path.join(save_dir, "teacher"),
            )[0]

            # synth dataset
            intersection_f1_macro_student = compute_per_intersection_macro_f1(
                {"0.5": self.decoded_student_05_buffer},
                self.hparams["data"]["test_tsv"],
                self.hparams["data"]["test_dur"],
            )

            # synth dataset
            intersection_f1_macro_teacher = compute_per_intersection_macro_f1(
                {"0.5": self.decoded_teacher_05_buffer},
                self.hparams["data"]["test_tsv"],
                self.hparams["data"]["test_dur"],
            )

            best_test_result = torch.tensor(max(psds1_student_psds_eval, psds2_student_psds_eval))

            results = {
                "hp_metric": best_test_result,
                "test/student/psds1_psds_eval": psds1_student_psds_eval,
                "test/student/psds1_sed_scores_eval": psds1_student_sed_scores_eval,
                "test/student/psds2_psds_eval": psds2_student_psds_eval,
                "test/student/psds2_sed_scores_eval": psds2_student_sed_scores_eval,
                "test/teacher/psds1_psds_eval": psds1_teacher_psds_eval,
                "test/teacher/psds1_sed_scores_eval": psds1_teacher_sed_scores_eval,
                "test/teacher/psds2_psds_eval": psds2_teacher_psds_eval,
                "test/teacher/psds2_sed_scores_eval": psds2_teacher_sed_scores_eval,
                "test/student/event_f1_macro": event_macro_student,
                "test/student/intersection_f1_macro": intersection_f1_macro_student,
                "test/teacher/event_f1_macro": event_macro_teacher,
                "test/teacher/intersection_f1_macro": intersection_f1_macro_teacher,
            }
        if self.logger is not None:
            self.logger.log_metrics(results)
            self.logger.log_hyperparams(self.hparams, results)

        for key in results.keys():
            self.log(key, results[key], prog_bar=True, logger=True)

    def configure_optimizers(self):
        return self.opt, self.scheduler

    def train_dataloader(self):

        self.train_loader = torch.utils.data.DataLoader(
            self.train_data,
            batch_sampler=self.train_sampler,
            num_workers=self.num_workers,
        )

        return self.train_loader

    def val_dataloader(self):
        self.val_loader = torch.utils.data.DataLoader(
            self.valid_data,
            batch_size=self.hparams["training"]["batch_size_val"],
            num_workers=self.num_workers,
            shuffle=False,
            drop_last=False,
        )
        return self.val_loader

    def test_dataloader(self):
        self.test_loader = torch.utils.data.DataLoader(
            self.test_data,
            batch_size=self.hparams["training"]["batch_size_val"],
            num_workers=self.num_workers,
            shuffle=False,
            drop_last=False,
        )
        return self.test_loader
