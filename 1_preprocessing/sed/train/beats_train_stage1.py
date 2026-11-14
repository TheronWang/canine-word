import argparse
import os
import sys
from time import sleep

import pandas as pd
import torch
import yaml
from torch.utils.data import ConcatDataset

sys.path.append('../')
sys.path.append('../../')

from desed_task.dataio.sampler import ConcatDatasetBatchSampler
from desed_task.utils.encoder import ManyHotEncoder
from desed_task.nnet.CRNN_beats import CRNN
from local.scheduler import ExponentialWarmup
from local.classes_dict import classes_labels
from train.local.beats_stage1_trainer import SEDTask4
from desed_task.dataio.datasets_beats_sed import StronglyAnnotatedSet, UnlabeledSet, WeakSet, StronglyValidationSet
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger
from collections import OrderedDict

import warnings

warnings.filterwarnings("ignore")

"""
conda activate atst_sed_dasheng
cd /home/xingyuan/animal_language2/code/ATST-SED2/train

CUDA_VISIBLE_DEVICES=1 python beats_train_stage1.py \
       --conf_file /home/xingyuan/animal_language2/code/ATST-SED2/train/confs/beats/stage1/base.yaml \
       --log_dir /home/xingyuan/animal_language2/code/ATST-SED2/work

CUDA_VISIBLE_DEVICES=0 nohup python beats_train_stage1.py \
       --conf_file /home/xingyuan/animal_language2/code/ATST-SED2/train/confs/beats/stage1/frozen.yaml \
       --log_dir /data/xingyuan/animal_language/v2/out/second_cut_model/beats/stage1 > /data/xingyuan/animal_language/v2/out/second_cut_model/beats/stage1/frozen.out 2>&1 &

CUDA_VISIBLE_DEVICES=0 nohup python beats_train_stage1.py \
       --conf_file /home/xingyuan/animal_language2/code/ATST-SED2/train/confs/beats/stage1/frozen_udesed.yaml \
       --log_dir /data/xingyuan/animal_language/v2/out/second_cut_model/beats/stage1 > /data/xingyuan/animal_language/v2/out/second_cut_model/beats/stage1/frozen_udesed.out 2>&1 &

"""

def separate_opt_params(model):
    # group parameters
    cnn_params = []
    rnn_params = []
    tfm_params = [[], [], [], [], [], [], [], [], [], [], [], [], [], []]
    for k, p in model.named_parameters():
        if "BEATs_model" not in k:
            if "cnn" in k:
                cnn_params.append(p)
            else:
                rnn_params.append(p)
        else:
            if "layers.0." in k:
                tfm_params[1].append(p)
            elif "layers.1." in k:
                tfm_params[2].append(p)
            elif "layers.2." in k:
                tfm_params[3].append(p)
            elif "layers.3." in k:
                tfm_params[4].append(p)
            elif "layers.4." in k:
                tfm_params[5].append(p)
            elif "layers.5." in k:
                tfm_params[6].append(p)
            elif "layers.6." in k:
                tfm_params[7].append(p)
            elif "layers.7." in k:
                tfm_params[8].append(p)
            elif "layers.8" in k:
                tfm_params[9].append(p)
            elif "layers.9." in k:
                tfm_params[10].append(p)
            elif "layers.10." in k:
                tfm_params[11].append(p)
            elif "layers.11." in k:
                tfm_params[12].append(p)
            elif ".layer_norm." in k:
                tfm_params[13].append(p)
            else:
                tfm_params[0].append(p)
    return cnn_params, rnn_params, list(reversed(tfm_params))


def get_sed_duration(path):
    sed_durations = {}
    with open(path, 'r') as file:
        duration_data = [line.strip().split() for line in file.readlines() if len(line.strip()) > 0]
    for dd in duration_data[1:]:
        assert dd[0] not in sed_durations.keys()
        sed_durations[dd[0]] = float(dd[1])
    return sed_durations


def interval_union(clips, interval_time=1e-3):
    if len(clips) < 2:
        return clips
    clips.sort(key=lambda c: c[0])
    start = clips[0][0]
    end = clips[0][1]

    new_clips = []
    for i in range(1, len(clips)):
        if (clips[i][0] - end) < interval_time:
            end = max(clips[i][1], end)
        else:
            new_clips.append([start, end, 'Dog'])
            start = clips[i][0]
            end = clips[i][1]
    new_clips.append([start, end, 'Dog'])
    return new_clips


def get_sed_ground_truth(path, audio_root, test_positive_labels):
    seds_ground_truth = {f: [] for f in os.listdir(audio_root)}
    with open(path, 'r') as file:
        gt_data = [line.strip().split() for line in file.readlines() if len(line.strip()) > 0]
    for gd in gt_data[1:]:
        if gd[3] in test_positive_labels:
            seds_ground_truth[gd[0]].append([float(gd[1]), float(gd[2]), 'Dog'])
    for key in seds_ground_truth.keys():
        seds_ground_truth[key] = interval_union(seds_ground_truth[key])
    return seds_ground_truth


def get_test_dataset(test_positive_labels, config):
    keys = ['chihuahua', 'husky', 'shiba_inu', 'pitbull', 'labrador', 'german_shepherd']
    tsv_root = '/home/xingyuan/animal_language2/data/dog_sed_dataset2/strong_valid_labels_tsv'
    audio_root = '/home/xingyuan/animal_language2/data/dog_sed_dataset2/strong_valid_audios_16k'

    seds_duration_dict = {key: {} for key in keys}
    seds_duration_dict['all'] = {}
    seds_ground_truth_dict = {key: {} for key in keys}
    seds_ground_truth_dict['all'] = {}
    dataset_list = []

    for key in keys:
        cur_duration_path = os.path.join(tsv_root, f'{key}_duration.tsv')
        cur_tsv_path = os.path.join(tsv_root, f'{key}.tsv')
        cur_audio_root = os.path.join(audio_root, key)

        cur_seds_durations = get_sed_duration(path=cur_duration_path)
        cur_seds_ground_truth = get_sed_ground_truth(path=cur_tsv_path,
                                                     audio_root=cur_audio_root,
                                                     test_positive_labels=test_positive_labels)
        cur_dataset = StronglyValidationSet(audio_root=cur_audio_root,
                                            seds_ground_truth0=cur_seds_ground_truth,
                                            frame_hop=config['feats']['hop_length'],
                                            net_pooling=config['data']['net_subsample'],
                                            fs=config['data']['fs'])

        seds_duration_dict[key] = cur_seds_durations
        for k, v in cur_seds_durations.items():
            assert k not in seds_duration_dict['all'].keys()
            seds_duration_dict['all'][k] = v
        seds_ground_truth_dict[key] = cur_seds_ground_truth
        for k, v in cur_seds_ground_truth.items():
            assert k not in seds_ground_truth_dict['all'].keys()
            seds_ground_truth_dict['all'][k] = v
        dataset_list.append(cur_dataset)
    return seds_duration_dict, seds_ground_truth_dict, ConcatDataset(dataset_list)


def get_train_strong_data(encoder, train_positive_labels, config):
    # # 旧的标注数据
    # old_tsv_path = "/home/xingyuan/animal_language2/data/dog_sed_dataset/youtube_strong_train_w10_s5.tsv"
    # old_audio_root = "/home/xingyuan/animal_language2/data/dog_sed_dataset/youtube_strong_train_audios_w10_s5_16k"
    # old_strong_df = pd.read_csv(old_tsv_path, sep="\t")
    # old_strong_set = StronglyAnnotatedSet(
    #     old_audio_root,
    #     old_strong_df,
    #     encoder,
    #     train_positive_labels=train_positive_labels,
    #     return_filename=True,
    #     pad_to=config["data"]["audio_max_len"],
    #     feat_params=config["feats"]
    # )
    # dataset_list = [old_strong_set]
    dataset_list = []

    # 新的标注数据
    keys = ['chihuahua', 'husky', 'shiba_inu', 'pitbull', 'labrador', 'german_shepherd']
    tsv_root = '/home/xingyuan/animal_language2/data/dog_sed_dataset2/strong_train_labels_tsv'
    audio_root = '/home/xingyuan/animal_language2/data/dog_sed_dataset2/strong_train_audios_16k_w10_s5'

    for key in keys:
        cur_tsv_path = os.path.join(tsv_root, f'{key}_w10_s5.tsv')
        cur_audio_root = os.path.join(audio_root, key)
        cur_strong_df = pd.read_csv(cur_tsv_path, sep="\t")
        cur_strong_set = StronglyAnnotatedSet(
            cur_audio_root,
            cur_strong_df,
            encoder,
            train_positive_labels=train_positive_labels,
            return_filename=True,
            pad_to=config["data"]["audio_max_len"],
            feat_params=config["feats"]
        )
        dataset_list.append(cur_strong_set)
    assert len(dataset_list) == len(keys) == 6
    return ConcatDataset(dataset_list)


def single_run(
        config,
        log_dir,
        gpus,
        checkpoint_resume=None,
        test_state_dict=None,
        fast_dev_run=False,
        evaluation=False,
        external=False,
        callbacks=None
):
    config.update({"log_dir": log_dir})

    # handle seed
    seed = config["training"]["seed"]
    if seed:
        pl.seed_everything(seed, workers=True)

    ##### data prep test ##########
    encoder = ManyHotEncoder(
        list(classes_labels.keys()),
        audio_len=config["data"]["audio_max_len"],
        frame_len=config["feats"]["n_filters"],
        frame_hop=config["feats"]["hop_length"],
        net_pooling=config["data"]["net_subsample"],
        fs=config["data"]["fs"],
    )

    # TODO: evaluation = False, 准备好测试集
    # use_audioset_flag = True
    train_positive_labels = 'Dog,dog,dog_dog,dog_noise'.split(',')
    test_positive_labels = 'Dog,dog,dog_dog'.split(',')
    if not evaluation:
        # devtest_df = pd.read_csv(config["data"]["test_tsv"], sep="\t")
        # devtest_dataset = StronglyAnnotatedSet(
        #     config["data"]["test_folder"],
        #     devtest_df,
        #     encoder,
        #     return_filename=True,
        #     pad_to=config["data"]["audio_max_len"],
        #     feat_params=config["feats"]
        # )

        seds_duration_dict, seds_ground_truth_dict, devtest_dataset = get_test_dataset(test_positive_labels, config)

    else:
        devtest_dataset = UnlabeledSet(
            config["data"]["eval_folder"],
            encoder,
            pad_to=None,
            return_filename=True,
            feat_params=config["feats"]
        )

    test_dataset = devtest_dataset

    ##### model definition  ############

    sed_student = CRNN(
        unfreeze_atst_layer=config["opt"]["tfm_trainable_layers"],
        **config["net"],
        model_init=config["ultra"]["model_init"],
        beats_init=config["ultra"]["beats_init"],
    )

    sed_teacher = CRNN(
        unfreeze_atst_layer=config["opt"]["tfm_trainable_layers"],
        **config["net"],
        model_init=config["ultra"]["model_init"],
        beats_init=config["ultra"]["beats_init"],
    )

    # sed_student.to('cuda:0')
    # sed_teacher.to('cuda:0')
    # sleep(60)
    # exit()

    # TODO: test_state_dict is None，准备训练集
    if test_state_dict is None:
        ##### data prep train valid ##########
        synth_df = pd.read_csv(config["data"]["synth_tsv"], sep="\t")
        synth_set = StronglyAnnotatedSet(
            config["data"]["synth_folder"],
            synth_df,
            encoder,
            train_positive_labels=train_positive_labels,
            return_filename=True,
            pad_to=config["data"]["audio_max_len"],
            feat_params=config["feats"]
        )

        strong_set = get_train_strong_data(encoder, train_positive_labels, config)

        weak_df = pd.read_csv(config["data"]["weak_tsv"], sep="\t")
        train_weak_df = weak_df.sample(
            frac=config["training"]["weak_split"],
            random_state=config["training"]["seed"],
        )
        valid_weak_df = weak_df.drop(train_weak_df.index).reset_index(drop=True)
        train_weak_df = train_weak_df.reset_index(drop=True)
        weak_set = WeakSet(
            config["data"]["weak_folder"],
            train_weak_df,
            encoder,
            return_filename=True,
            pad_to=config["data"]["audio_max_len"],
            feat_params=config["feats"]
        )
        # synth_df_val = pd.read_csv(config["data"]["synth_val_tsv"], sep="\t")
        # synth_val = StronglyAnnotatedSet(
        #     config["data"]["synth_val_folder"],
        #     synth_df_val,
        #     encoder,
        #     return_filename=True,
        #     pad_to=config["data"]["audio_max_len"],
        #     feat_params=config["feats"]
        # )
        unlabeled_set = UnlabeledSet(
            config["data"]["unlabeled_folder"],
            encoder,
            return_filename=True,
            pad_to=config["data"]["audio_max_len"],
            feat_params=config["feats"]
        )
        # strong_df_val = pd.read_csv(config["data"]["strong_val_tsv"], sep="\t")
        # strong_val = StronglyAnnotatedSet(
        #     config["data"]["strong_folder"],
        #     strong_df_val,
        #     encoder,
        #     return_filename=True,
        #     pad_to=config["data"]["audio_max_len"],
        #     feat_params=config["feats"]
        # )
        # weak_val = WeakSet(
        #     config["data"]["weak_folder"],
        #     valid_weak_df,
        #     encoder,
        #     pad_to=config["data"]["audio_max_len"],
        #     return_filename=True,
        #     feat_params=config["feats"]
        # )

        # weak set * 20
        weak_set = ConcatDataset([weak_set for _ in range(20)])
        if len(config["training"]["batch_size"]) == 4:
            tot_train_data = [strong_set, synth_set, weak_set, unlabeled_set]
        else:
            tot_train_data = [strong_set, weak_set, unlabeled_set]
        train_dataset = torch.utils.data.ConcatDataset(tot_train_data)
        assert len(unlabeled_set) == 14412
        for i in range(len(tot_train_data)):
            print(f'{i}: {len(tot_train_data[i])}')
        print()

        batch_sizes = config["training"]["batch_size"]
        samplers = [torch.utils.data.RandomSampler(x) for x in tot_train_data]
        batch_sampler = ConcatDatasetBatchSampler(samplers, batch_sizes)
        # valid_dataset = torch.utils.data.ConcatDataset([weak_val, synth_val, strong_val])

        ##### training params and optimizers ############
        epoch_len = min(
            [
                len(tot_train_data[indx])
                // (
                        config["training"]["batch_size"][indx]
                        * config["training"]["accumulate_batches"]
                )
                for indx in range(len(tot_train_data))
            ]
        )

        cnn_params, rnn_params, tfm_params = separate_opt_params(sed_student)

        cnn_param_groups = [
            {"params": cnn_params, "lr": config["opt"]["cnn_lr"]}
        ]
        rnn_param_groups = [
            {"params": rnn_params, "lr": config["opt"]["rnn_lr"]},
        ]
        tfm_trainable_params = []
        trainable_layers = config["opt"]["tfm_trainable_layers"]
        for i in range(trainable_layers):
            tfm_trainable_params.append(tfm_params[i])
            for p in tfm_params[i]:
                p.requires_grad = True

        init_lr = config["opt"]["tfm_lr"]
        lr_scale = config["opt"]["tfm_lr_scale"]
        scale_lrs = [init_lr * (lr_scale ** i) for i in range(trainable_layers)]
        tfm_param_groups = [{"params": tfm_trainable_params[i], "lr": scale_lrs[i]} for i in
                            range(len(tfm_trainable_params))]
        max_lrs = [cnn_param_groups[0]["lr"]] + [rnn_param_groups[0]["lr"]] + [x["lr"] for x in tfm_param_groups]
        param_groups = cnn_param_groups + rnn_param_groups + tfm_param_groups
        opt = torch.optim.Adam(param_groups, betas=(0.9, 0.999))
        opt = [opt]
        exp_steps = config["training"]["n_epochs_warmup"] * epoch_len
        tot_steps = config["training"]["n_epochs"] * epoch_len

        exp_scheduler = {
            "scheduler": ExponentialWarmup(opt[0], max_lrs, exp_steps, tot_steps=tot_steps, cos_down=False),
            "interval": "step",
        }
        exp_scheduler = [exp_scheduler]

        # init opt
        init_lrs = exp_scheduler[0]["scheduler"]._get_lr()
        for i, lr in enumerate(init_lrs):
            opt[0].param_groups[i]["lr"] = lr

        # logger = TensorBoardLogger(
        #     os.path.dirname(config["log_dir"]), config["log_dir"].split("/")[-1],
        # )
        logger = CSVLogger(os.path.dirname(config["log_dir"]), config["log_dir"].split("/")[-1])
        logger.log_hyperparams(config)
        print(f"experiment dir: {logger.log_dir}")

        if callbacks is None:
            callbacks = [
                # EarlyStopping(
                #     monitor="val/obj_metric",
                #     patience=config["training"]["early_stop_patience"],
                #     verbose=True,
                #     mode="min",
                # ),
                ModelCheckpoint(
                    os.path.join(logger.log_dir, 'checkpoints'),
                    # monitor="val/obj_metric",
                    save_top_k=-1,
                    # mode="min",
                    save_last=True,
                    every_n_train_steps=28,
                ),
            ]
    else:
        train_dataset = None
        valid_dataset = None
        batch_sampler = None
        opt = None
        exp_scheduler = None
        logger = True
        callbacks = None

    desed_training = SEDTask4(
        config,
        encoder=encoder,
        sed_student=sed_student,
        opt=opt,
        train_data=train_dataset,
        # valid_data=valid_dataset,
        valid_data=test_dataset,
        test_data=test_dataset,
        train_sampler=batch_sampler,
        scheduler=exp_scheduler,
        fast_dev_run=fast_dev_run,
        evaluation=evaluation,

        sed_teacher=sed_teacher,
        seds_duration_dict=seds_duration_dict,
        seds_ground_truth_dict=seds_ground_truth_dict,
        valid_results_save_path=os.path.join(logger.log_dir, 'valid_results.jsonlines')
    )

    # Not using the fast_dev_run of Trainer because creates a DummyLogger so cannot check problems with the Logger
    if fast_dev_run:
        log_every_n_steps = 1
        limit_train_batches = 2
        limit_val_batches = 1.
        limit_test_batches = 1.
        n_epochs = 3
    else:
        log_every_n_steps = 10
        limit_train_batches = 1.0
        limit_val_batches = 1.0
        limit_test_batches = 1.0
        n_epochs = config["training"]["n_epochs"]

    accelerator = "gpu"
    devices = gpus

    trainer = pl.Trainer(
        precision=config["training"]["precision"],
        max_epochs=n_epochs,
        callbacks=callbacks,
        accelerator=accelerator,
        devices=devices,
        strategy=config["training"].get("backend"),
        accumulate_grad_batches=config["training"]["accumulate_batches"],
        logger=logger,
        gradient_clip_val=config["training"]["gradient_clip"],
        val_check_interval=config["training"]["validation_interval"],
        num_sanity_val_steps=0,
        log_every_n_steps=log_every_n_steps,
        limit_train_batches=limit_train_batches,
        limit_val_batches=limit_val_batches,
        limit_test_batches=limit_test_batches,
        deterministic=config["training"]["deterministic"],
        enable_progress_bar=config["training"]["enable_progress_bar"],
    )
    if test_state_dict is None:
        # start tracking energy consumption
        trainer.fit(desed_training, ckpt_path=checkpoint_resume)
        best_path = trainer.checkpoint_callback.best_model_path
        print(f"best model: {best_path}")
        test_state_dict = torch.load(best_path)["state_dict"]

    desed_training.load_state_dict(test_state_dict)
    trainer.test(desed_training)


def prepare_run(argv=None):
    parser = argparse.ArgumentParser("Training a SED system for DESED Task")
    parser.add_argument(
        "--conf_file",
        default="./confs/stage1_real.yaml",
        help="The configuration file with all the experiment parameters.",
    )
    parser.add_argument(
        "--log_dir",
        default="./exp/stage1/",
        help="Directory where to save tensorboard logs, saved models, etc.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        default=None,
        help="Allow the training to be resumed, take as input a previously saved model (.ckpt).",
    )
    parser.add_argument(
        "--test_from_checkpoint", default=None, help="Test the model specified"
    )
    parser.add_argument(
        "--gpus",
        default="1",
        help="The number of GPUs to train on, or the gpu to use, default='0', "
             "so uses one GPU",
    )
    parser.add_argument(
        "--fast_dev_run",
        action="store_true",
        default=False,
        help="Use this option to make a 'fake' run which is useful for development and debugging. "
             "It uses very few batches and epochs so it won't give any meaningful result.",
    )
    parser.add_argument(
        "--eval_from_checkpoint",
        default=None,
        help="Evaluate the model specified"
    )
    parser.add_argument(
        "--external",
        action="store_true",
        default=False,

    )
    parser.add_argument(
        "--prefix",
        default="",
        type=str,
        help="The strong annotations coming from Audioset will be included in the training phase.",
    )

    args = parser.parse_args(argv)
    args.log_dir += args.prefix
    with open(args.conf_file, "r") as f:
        configs = yaml.safe_load(f)

    evaluation = False
    test_from_checkpoint = args.test_from_checkpoint

    if args.eval_from_checkpoint is not None:
        test_from_checkpoint = args.eval_from_checkpoint
        evaluation = True

    test_model_state_dict = None
    if test_from_checkpoint is not None:
        checkpoint = torch.load(test_from_checkpoint)
        configs_ckpt = checkpoint["hyper_parameters"]
        configs_ckpt["data"] = configs["data"]
        print(
            f"loaded model: {test_from_checkpoint} \n"
            f"at epoch: {checkpoint['epoch']}"
        )
        test_model_state_dict = checkpoint["state_dict"]
        configs = configs_ckpt

    if evaluation:
        configs["training"]["batch_size_val"] = 1

    test_only = test_from_checkpoint is not None
    return configs, args, test_model_state_dict, evaluation


if __name__ == "__main__":
    torch.set_float32_matmul_precision('high')
    # prepare run
    configs, args, test_model_state_dict, evaluation = prepare_run()

    # launch run
    single_run(
        configs,
        args.log_dir,
        args.gpus,
        args.resume_from_checkpoint,
        test_model_state_dict,
        args.fast_dev_run,
        evaluation,
        args.external
    )
