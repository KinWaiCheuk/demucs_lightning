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

# #library for class Wavset
# from collections import OrderedDict
# import math
# import torch as th
# import julius
# from torch.nn import functional as F

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
    train_set = MusdbHQ(root = args.dset.train.root, 
                        subset= 'training', 
                        download = args.dset.train.download, 
                        segment= args.dset.train.segment, 
                        shift= args.dset.train.shift, 
                        normalize= args.dset.train.normalize,
                        samplerate= args.dset.train.samplerate, 
                        channels= args.dset.train.channels, 
                        ext= args.dset.train.ext)

    valid_set = MusdbHQ(root = args.dset.valid.root, 
                        subset= 'validation', 
                        download = args.dset.valid.download, 
                        segment= args.dset.valid.segment, 
                        shift= args.dset.valid.shift, 
                        normalize= args.dset.valid.normalize,
                        samplerate= args.dset.valid.samplerate, 
                        channels= args.dset.valid.channels, 
                        ext= args.dset.valid.ext)
    
    test_set = MusdbHQ(root = args.dset.test.root, 
                       subset= 'test', 
                       download = args.dset.test.download, 
                       segment= args.dset.test.segment, 
                       shift= args.dset.test.shift, 
                       normalize= args.dset.test.normalize,
                       samplerate= args.dset.test.samplerate, 
                       channels= args.dset.test.channels, 
                       ext= args.dset.test.ext)

    train_loader = DataLoader(train_set, 
                              batch_size=args.dataloader.train.batch_size, 
                              shuffle= args.dataloader.train.shuffle,
                              num_workers=args.dataloader.train.num_workers, drop_last=True)

    valid_loader = DataLoader(valid_set, 
                              batch_size=args.dataloader.valid.batch_size, 
                              shuffle= args.dataloader.valid.shuffle,
                              num_workers=args.dataloader.valid.num_workers, drop_last=False)
    
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
    
#     print(f'optimizer = {model.optimizers}')
    
#     print(f'len train_set= {len(train_set)}') #len train_set= 18368
#     print(f'len valid_set= {len(valid_set)}') #len valid_set= 14
    
    checkpoint_callback = ModelCheckpoint(**args.checkpoint,auto_insert_metric_name=False)
    #auto_insert_metric_name = False: won't refer the '/' in filename as path

    name = f'{args.model}_experiment_epoch={args.epochs}_augmentation={args.data_augmentation}'
    #file name shown in tensorboard logger

    lr_monitor = LearningRateMonitor(logging_interval='step')
    logger = TensorBoardLogger(save_dir=".", version=1, name=name)
    trainer = pl.Trainer(gpus=args.gpus,
                         max_epochs=args.epochs,
                         callbacks=[checkpoint_callback, lr_monitor],
                         logger=logger,
                         check_val_every_n_epoch=1)


    trainer.fit(model, train_loader,valid_loader)
    trainer.test(model, test_loader)
    
if __name__ == "__main__":
    main()      