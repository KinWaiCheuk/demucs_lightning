import torch
from demucs.svd import svd_penalty
from hydra import initialize, compose
from demucs.demucs import Demucs
from demucs.hdemucs import HDemucs
from omegaconf import OmegaConf

# for dataset
import hashlib
import hydra
from pathlib import Path
import json
import os
import tqdm
import torchaudio as ta
import musdb

#library for class Wavset
from collections import OrderedDict
import math
import torch as th
import julius
from torch.nn import functional as F

#library for loader()
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader, Subset


import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
import sys
from demucs.states import get_quantizer

# =====================Functions for dataset===========================

# This part can be refactors into a single class MusdbHQ
        
EXT = ".wav"        
MIXTURE = "mixture"

def convert_audio_channels(wav, channels=2):
    """Convert audio to the given number of channels."""
    *shape, src_channels, length = wav.shape
    if src_channels == channels:
        pass
    elif channels == 1:
        # Case 1:
        # The caller asked 1-channel audio, but the stream have multiple
        # channels, downmix all channels.
        wav = wav.mean(dim=-2, keepdim=True)
    elif src_channels == 1:
        # Case 2:
        # The caller asked for multiple channels, but the input file have
        # one single channel, replicate the audio over all channels.
        wav = wav.expand(*shape, channels, length)
    elif src_channels >= channels:
        # Case 3:
        # The caller asked for multiple channels, and the input file have
        # more channels than requested. In that case return the first channels.
        wav = wav[..., :channels, :]
    else:
        # Case 4: What is a reasonable choice here?
        raise ValueError('The audio file has less channels than requested but is not mono.')
    return wav

class Wavset:
    def __init__(
            self,
            root, metadata, sources,
            segment=None, shift=None, normalize=True,
            samplerate=44100, channels=2, ext=EXT):
        """
        Waveset (or mp3 set for that matter). Can be used to train
        with arbitrary sources. Each track should be one folder inside of `path`.
        The folder should contain files named `{source}.{ext}`.
        Args:
            root (Path or str): root folder for the dataset.
            metadata (dict): output from `build_metadata`.
            sources (list[str]): list of source names.
            segment (None or float): segment length in seconds. If `None`, returns entire tracks.
            shift (None or float): stride in seconds bewteen samples.
            normalize (bool): normalizes input audio, **based on the metadata content**,
                i.e. the entire track is normalized, not individual extracts.
            samplerate (int): target sample rate. if the file sample rate
                is different, it will be resampled on the fly.
            channels (int): target nb of channels. if different, will be
                changed onthe fly.
            ext (str): extension for audio files (default is .wav).
        samplerate and channels are converted on the fly.
        """
        self.root = Path(root)
        self.metadata = OrderedDict(metadata)
        self.segment = segment
        self.shift = shift or segment
        self.normalize = normalize
        self.sources = sources
        self.channels = channels
        self.samplerate = samplerate
        self.ext = ext
        self.num_examples = []
        for name, meta in self.metadata.items():
            track_duration = meta['length'] / meta['samplerate']
            if segment is None or track_duration < segment:
                examples = 1
            else:
                examples = int(math.ceil((track_duration - self.segment) / self.shift) + 1)
            self.num_examples.append(examples)

    def __len__(self):
        return sum(self.num_examples)

    def get_file(self, name, source):
        return self.root / name / f"{source}{self.ext}"

    def __getitem__(self, index):
        for name, examples in zip(self.metadata, self.num_examples):           
            if index >= examples:
                index -= examples
                continue
            meta = self.metadata[name]
            num_frames = -1
            offset = 0
            if self.segment is not None:
                offset = int(meta['samplerate'] * self.shift * index)
                num_frames = int(math.ceil(meta['samplerate'] * self.segment))
            wavs = []
            for source in self.sources:
                file = self.get_file(name, source)
                wav, _ = ta.load(str(file), frame_offset=offset, num_frames=num_frames)
                wav_before = wav
                wav = convert_audio_channels(wav, self.channels)
                wavs.append(wav)

            example = th.stack(wavs)
            example = julius.resample_frac(example, meta['samplerate'], self.samplerate)
            if self.normalize:
                example = (example - meta['mean']) / meta['std']
            if self.segment:
                length = int(self.segment * self.samplerate)
                example = example[..., :length]
                example = F.pad(example, (0, length - example.shape[-1]))
            return example

def _get_musdb_valid():
    # Return musdb valid set.
    import yaml
    setup_path = Path(musdb.__path__[0]) / 'configs' / 'mus.yaml'
    setup = yaml.safe_load(open(setup_path, 'r'))
    return setup['validation_tracks']

def build_metadata(path, sources, normalize=True, ext=EXT):
    """
    Build the metadata for `Wavset`.
    Args:
        path (str or Path): path to dataset.
        sources (list[str]): list of sources to look for.
        normalize (bool): if True, loads full track and store normalization
            values based on the mixture file.
        ext (str): extension of audio files (default is .wav).
    """

    meta = {}
    path = Path(path)
    pendings = []
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(8) as pool:
        for root, folders, files in os.walk(path, followlinks=True):
            root = Path(root)
            if root.name.startswith('.') or folders or root == path:
                continue
            name = str(root.relative_to(path))
            pendings.append((name, pool.submit(_track_metadata, root, sources, normalize, ext)))
            # meta[name] = _track_metadata(root, sources, normalize, ext)
        for name, pending in tqdm.tqdm(pendings, ncols=120):
            meta[name] = pending.result()
    return meta

def _track_metadata(track, sources, normalize=True, ext=EXT):
    track_length = None
    track_samplerate = None
    mean = 0
    std = 1
    for source in sources + [MIXTURE]:
        file = track / f"{source}{ext}"
        try:
            info = ta.info(str(file))
        except RuntimeError:
            print(file)
            raise
        length = info.num_frames
        if track_length is None:
            track_length = length
            track_samplerate = info.sample_rate
        elif track_length != length:
            raise ValueError(
                f"Invalid length for file {file}: "
                f"expecting {track_length} but got {length}.")
        elif info.sample_rate != track_samplerate:
            raise ValueError(
                f"Invalid sample rate for file {file}: "
                f"expecting {track_samplerate} but got {info.sample_rate}.")
        if source == MIXTURE and normalize:
            try:
                wav, _ = ta.load(str(file))
            except RuntimeError:
                print(file)
                raise
            wav = wav.mean(0)
            mean = wav.mean().item()
            std = wav.std().item()

    return {"length": length, "mean": mean, "std": std, "samplerate": track_samplerate}

# =====================End of dataset functions===========================

@hydra.main(config_path="conf", config_name="config")
def main(args):  
    train_valid=False
    full_cv=True
    """Extract the musdb dataset from the XP arguments."""
    sig = hashlib.sha1(str(args.dset.musdb).encode()).hexdigest()[:8]
    metadata_file = Path('./metadata') / ('musdb_' + sig + ".json")
    root = Path(args.dset.musdb) / "train"
    #     if not metadata_file.is_file() and distrib.rank == 0:
    if not metadata_file.is_file():
        metadata_file.parent.mkdir(exist_ok=True, parents=True)
        metadata = build_metadata(root, args.dset.sources)
        json.dump(metadata, open(metadata_file, "w"))
    #     if distrib.world_size > 1:
    #         distributed.barrier()
    metadata = json.load(open(metadata_file))

    valid_tracks = _get_musdb_valid()

    if train_valid:
        metadata_train = metadata
    else:
        metadata_train = {name: meta for name, meta in metadata.items() if name not in valid_tracks}
    metadata_valid = {name: meta for name, meta in metadata.items() if name in valid_tracks}
    if full_cv:
        kw_cv = {}
    else:
        kw_cv = {'segment': segment, 'shift': shift}



    train_set = Wavset(root, metadata_train, args.dset.sources,
                       segment=args.dset.segment, shift=args.dset.shift,
                       samplerate=args.dset.samplerate, channels=args.dset.channels,
                       normalize=args.dset.normalize)
    
#     valid_sources =  [MIXTURE] + list(args.dset.sources)
    valid_set = Wavset(root, metadata_valid, [MIXTURE] + list(args.dset.sources),                      
                       samplerate=args.dset.samplerate,
                       channels=args.dset.channels,
                       normalize=args.dset.normalize, **kw_cv)
    
    train_loader = DataLoader(
            train_set, batch_size=args.train_batch_size, shuffle=True,
            num_workers=args.misc.num_workers, drop_last=True)

    valid_loader = DataLoader(
            valid_set, batch_size=args.valid_batch_size, shuffle=True,
            num_workers=args.misc.num_workers, drop_last=True)
    
    model = Demucs(
                   sources=args.dset.sources,
                   audio_channels=args.dset.channels,
                   samplerate=args.dset.samplerate,
                   segment=4 * args.dset.segment,
                   **args.demucs,
                   args=args
                  )
    quantizer = get_quantizer(model, args.quant, model.optimizers)
    model.quantizer = quantizer #can use as self.quantizer in class Demucs
    
#     print(f'optimizer = {model.optimizers}')
    
#     print(f'len train_set= {len(train_set)}') #len train_set= 18368
#     print(f'len valid_set= {len(valid_set)}') #len valid_set= 14
    
    checkpoint_callback = ModelCheckpoint(**args.checkpoint,auto_insert_metric_name=False)
    #auto_insert_metric_name = False: won't refer the '/' in filename as path

    name = f'demucs_experiment'
    #file name shown in tensorboard logger

    lr_monitor = LearningRateMonitor(logging_interval='step')
    logger = TensorBoardLogger(save_dir=".", version=1, name=name)
    trainer = pl.Trainer(gpus=args.gpus,
                         max_epochs=args.epochs,
                         callbacks=[checkpoint_callback, lr_monitor],
                         logger=logger,
                         check_val_every_n_epoch=1)


    trainer.fit(model, train_loader,valid_loader)
    
if __name__ == "__main__":
    main()      