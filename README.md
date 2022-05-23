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
`Python==3.8.10` and `ffmpeg` is required to run this repo.

If `ffmpeg` is not installed on your machine, you can install it via `apt install ffmpeg`

You can install all required libraries at once via
``` bash
pip install -r requirements.txt
```

## Train the model
If it is your first time running the repo, you can use the argument `download=True` to automatically download and setup the `musdb18hq` dataset.

```bash
python train.py gpus=[0] data_root='./' model=HDemucs download=True
```

### arguments
The full list of arguments and their default values can be found in `conf/config.yaml`.

__gpus__: Select which GPU to use. If you have multiple GPUs on your machine and you want to use GPU:2, you can set `gpus=[2]`. If you want to use DDP (multi-GPU training), you can set `gpus=2`, it will automatically use the first two GPUs avaliable in your machine. If you want to use GPU:0, GPU:2, and GPU:3 for training, you can set `gpus=[0,2,3]`.

__data_root__: Select the location of your dataset. If `download=True`, it will become the directory that the dataset is going to be downloaded to.

__model__: Select which version of demucs to use. Default model of this repo is Hybrid Demucs (v3). You can switch to Demucs (v2) by setting the `model=Demucs`.

__samplerate__: The sampling rate for the audio. Default as `44100`.

__epochs__: The number of epochs to train the model. Default as `360`.

__optim.lr__: Learning rate of the optimizer. Default as `3e-4`.

For detail information of the configuration, you can check via `conf/config.yaml`