# for dataset
import hashlib
import json
import os
import sys
from pathlib import Path

import hydra
import pytorch_lightning as pl
import torch
import torchaudio as ta
import tqdm

# library for Musdb dataset
from AudioLoader.music.mss import MusdbHQ
from hydra import compose, initialize
from hydra.utils import to_absolute_path
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from torch.utils.data import DataLoader, Subset

# library for loader()
from torch.utils.data.distributed import DistributedSampler

from demucs.demucs import Demucs
from demucs.hdemucs import HDemucs
from demucs.states import get_quantizer
from demucs.svd import svd_penalty


@hydra.main(config_path="conf", config_name="train_test_config")
def main(args):
    args.data_root = to_absolute_path(args.data_root)

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
        model = model.load_from_checkpoint(to_absolute_path(args.resume_checkpoint))

    elif args.model == "HDemucs":
        model = HDemucs(
            sources=args.sources,
            samplerate=args.samplerate,
            segment=4 * args.dset.train.segment,
            **args.hdemucs,
            args=args,
        )
        model = model.load_from_checkpoint(to_absolute_path(args.resume_checkpoint))

    else:
        print("Invalid model, please choose Demucs or HDemucs")

    quantizer = get_quantizer(model, args.quant, model.optimizers)
    model.quantizer = quantizer  # can use as self.quantizer in class Demucs

    name = f"Testing_{args.checkpoint.filename}"
    # file name shown in tensorboard logger

    lr_monitor = LearningRateMonitor(logging_interval="step")

    if args.logger == "tensorboard":
        logger = TensorBoardLogger(save_dir=".", version=1, name=name)
    elif args.logger == "wandb":
        logger = WandbLogger(project="demucs_lightning_test", **args.wandb)
    else:
        raise Exception(f"Logger {args.logger} not implemented")

    trainer = pl.Trainer(**args.trainer, logger=logger)

    trainer.test(model, test_loader)


if __name__ == "__main__":
    main()
