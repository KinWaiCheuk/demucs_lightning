import torch
from demucs.svd import svd_penalty
from hydra import initialize, compose
from hydra.utils import to_absolute_path
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


#library for loader()
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader, Subset


import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
import sys
from demucs.states import get_quantizer

#library for Musdb dataset
from AudioLoader.music.mss import MusdbHQ


@hydra.main(config_path="conf", config_name="config")
def main(args):    
    args.data_root = to_absolute_path(args.data_root) 
    
    test_set = MusdbHQ(root = args.dset.test.root, 
                       subset= 'test', 
                       download = args.dset.test.download, 
                       segment= args.dset.test.segment, 
                       shift= args.dset.test.shift, 
                       normalize= args.dset.test.normalize,
                       samplerate= args.dset.test.samplerate, 
                       channels= args.dset.test.channels, 
                       ext= args.dset.test.ext)

    
    test_loader = DataLoader(test_set, 
                             batch_size=args.dataloader.test.batch_size, 
                             shuffle= args.dataloader.test.shuffle,
                             num_workers=args.dataloader.test.num_workers, drop_last=False)
    
    if args.model == 'Demucs':
        model = Demucs(sources=args.dset.sources,                    
                       samplerate=args.samplerate,
                       segment=4 * args.dset.train.segment,
                       **args.demucs,
                       args=args)
    
    elif args.model == 'HDemucs':
        model = HDemucs(sources=args.dset.sources,
                        samplerate=args.samplerate,
                        segment=4 * args.dset.train.segment,
                        **args.hdemucs,
                        args=args)
                        
    else:
        print('Invalid model, please choose Demucs or HDemucs')
        
    quantizer = get_quantizer(model, args.quant, model.optimizers)
    model.quantizer = quantizer #can use as self.quantizer in class Demucs
    
    model = model.load_from_checkpoint(to_absolute_path(args.resume_checkpoint),
                               sources=args.dset.sources,
                               samplerate=args.samplerate,
                               segment=4 * args.dset.train.segment,
                               **args.hdemucs,
                               args=args)
    

    name = f'Testing_{args.checkpoint.filename}'
    #file name shown in tensorboard logger

    lr_monitor = LearningRateMonitor(logging_interval='step')
    logger = TensorBoardLogger(save_dir=".", version=1, name=name)

    trainer = pl.Trainer(**args.trainer,
                         logger=logger)


    trainer.test(model, test_loader)
    
if __name__ == "__main__":
    main()      