# Welcome to Demucs Music Source Separation with lightning module

1. [Introduction](#Introduction)
1. [Requirement](#Requirement)
1. [Train the model](#Train-the-model)
1. [Key default setting](#Key-default-setting)


## Introduction
```
demucs_lightning
├── conf
│     ├─config.yaml
│     │
│
├──demucs
│     ├─demucs.py
│     ├─hdemucs.py
│     ├─other custome modules
│
├──train.py
│   
```

There are 2 major released version of Demucs. 
* Demucs (v2) used waveform as domain. 
* Hybrid Demucs (v3) is featuring hybrid source separation. 

You can find their model structure in 
`demucs.py` and `hdemucs.py` from demucs folder.\
For the official information of Demucs, you can visit [facebookresearch/demucs](https://github.com/facebookresearch/demucs)

Demucs is trained by [MusdbHQ](https://sigsep.github.io/datasets/musdb.html). This repo uses `AudioLoader` to get MusdbHQ dataset .\
For more information of Audioloader, you can visit [KinWaiCheuk/AudioLoader](https://github.com/KinWaiCheuk/AudioLoader).

Or else you can download MusdbHQ dataset manually from [zenodo](https://zenodo.org/record/3338373#.YoEmSC8RpQI).

## Requirement

Python3.8 is required to run this repo.

You can install all required libraries at once via
``` bash
pip install -r requirements.txt
```

## Train the model
```bash
python train.py
```

Note:

* If this is your 1st time to train the model, you need to set `download` argument to True via

```bash
python train.py download=True
```
* If you have existing MusdbHQ dataset, you need to indicate its path in `data_root` in `conf/config.yaml`

* You can set which GPU to use by using the `gpus` argument.

``` bash
python train.py gpus=[1]
```
* Default model of this repo is Hybrid Demucs (v3). You can switch to Demucs (v2) by changing the `model` argument.

```bash
python train.py model=Demucs
```

## Key default setting
* model=HDemucs
* data_augmentation=True
* samplerate=44100
* For training set: `dset.train.segment=11` `dset.train.shift=1` `dataloader.train.batch_size=4`
* For validation set: `dset.valid.segment=None` `dset.valid.shift=None` `dataloader.valid.batch_size=1`
* For test set: `dset.test.segment=None` `dset.test.shift=None` `dataloader.test.batch_size=1`
* epochs=360
* optim.lr=3e-4

For detail information of the configuration, you can check via `conf/config.yaml`