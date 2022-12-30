# for dataset
import hashlib
import json
import os

# library for Musdb dataset
import sys
from pathlib import Path

import hydra
import pytorch_lightning as pl
import torch
import torchaudio as ta
import tqdm
from hydra import compose, initialize
from hydra.utils import to_absolute_path
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.strategies import DDPStrategy
from torch.utils.data import DataLoader, Subset

# library for loader()
from torch.utils.data.distributed import DistributedSampler

from demucs.demucs import Demucs
from demucs.hdemucs import HDemucs
from demucs.states import get_quantizer
from demucs.svd import svd_penalty

sys.path.insert(0, "/workspace/helen/AudioLoader")
from AudioLoader.music.mss import MusdbHQ


@hydra.main(config_path="conf", config_name="train_test_config")
def main(args):
    args.data_root = to_absolute_path(args.data_root)
    train_set = MusdbHQ(
        root=args.dset.train.root,
        subset="training",
        sources=["drums", "bass", "other", "vocals"],
        # have to be 4 sourcse, to make mix in training_step  #mix = sources.sum(dim=1)
        download=args.dset.train.download,
        segment=args.dset.train.segment,
        shift=args.dset.train.shift,
        normalize=args.dset.train.normalize,
        samplerate=args.dset.train.samplerate,
        channels=args.dset.train.channels,
        ext=args.dset.train.ext,
    )

    valid_set = MusdbHQ(
        root=args.dset.valid.root,
        subset="validation",
        sources=args.dset.valid.sources,
        download=args.dset.valid.download,
        segment=args.dset.valid.segment,
        shift=args.dset.valid.shift,
        normalize=args.dset.valid.normalize,
        samplerate=args.dset.valid.samplerate,
        channels=args.dset.valid.channels,
        ext=args.dset.valid.ext,
    )

    test_set = MusdbHQ(
        root=args.dset.test.root,
        subset="test",
        sources=args.dset.test.sources,
        download=args.dset.test.download,
        segment=args.dset.test.segment,
        shift=args.dset.test.shift,
        normalize=args.dset.test.normalize,
        samplerate=args.dset.test.samplerate,
        channels=args.dset.test.channels,
        ext=args.dset.test.ext,
    )

    train_loader = DataLoader(
        train_set,
        batch_size=args.dataloader.train.batch_size,
        shuffle=args.dataloader.train.shuffle,
        num_workers=args.dataloader.train.num_workers,
        drop_last=True,
    )

    valid_loader = DataLoader(
        valid_set,
        batch_size=args.dataloader.valid.batch_size,
        shuffle=args.dataloader.valid.shuffle,
        num_workers=args.dataloader.valid.num_workers,
        drop_last=False,
    )

    test_loader = DataLoader(
        test_set,
        batch_size=args.dataloader.test.batch_size,
        shuffle=args.dataloader.test.shuffle,
        num_workers=args.dataloader.test.num_workers,
        drop_last=False,
    )

    if args.model == "Demucs":
        model = Demucs(
            sources=args.sources,
            samplerate=args.samplerate,
            segment=4 * args.dset.train.segment,
            **args.demucs,
            args=args,
        )

    elif args.model == "HDemucs":
        model = HDemucs(
            sources=args.sources,
            samplerate=args.samplerate,
            segment=4 * args.dset.train.segment,
            **args.hdemucs,
            args=args,
        )

    else:
        print("Invalid model, please choose Demucs or HDemucs")

    quantizer = get_quantizer(model, args.quant, model.optimizers)
    model.quantizer = quantizer  # can use as self.quantizer in class Demucs

    #     print(f'optimizer = {model.optimizers}')

    #     print(f'len train_set= {len(train_set)}') #len train_set= 18368
    #     print(f'len valid_set= {len(valid_set)}') #len valid_set= 14

    checkpoint_callback = ModelCheckpoint(
        **args.checkpoint, auto_insert_metric_name=False
    )
    # auto_insert_metric_name = False: won't refer the '/' in filename as path

    name = f"{args.model}_experiment_epoch={args.epochs}_augmentation={args.data_augmentation}"
    # file name shown in tensorboard logger

    lr_monitor = LearningRateMonitor(logging_interval="step")
    logger = TensorBoardLogger(save_dir=".", version=1, name=name)
    if (
        args.trainer.resume_from_checkpoint
    ):  # resume previous training when this is given
        args.trainer.resume_from_checkpoint = to_absolute_path(
            args.trainer.resume_from_checkpoint
        )
        print(f"Resume training from {args.trainer.resume_from_checkpoint}")
    trainer = pl.Trainer(
        **args.trainer,
        callbacks=[checkpoint_callback, lr_monitor],
        strategy=DDPStrategy(find_unused_parameters=False),
        logger=logger,
    )

    trainer.fit(model, train_loader, valid_loader)
    trainer.test(model, test_loader)


if __name__ == "__main__":
    main()
