# Demucs lightning

1. [Introduction](#Introduction)
1. [Requirement](#Requirement)
1. [Training](#Training)
    1. [Demucs](#Demucs)
    1. [HDemucs](#HDemucs)
1. [Resume from Checkpoints](#Resume-from-Checkpoints)
1. [Training with a less powerful GPU](#Training-with-a-less-powerful-GPU)
1. [Testing pretrained model](#Testing-pretrained-model)
1. [Half-Precision Training](#Half-Precision-Training)
1. [Default settings](#Default-settings)
1. [Inferencing](#Inferencing)
1. [Development](#Development)


## Introduction
```
demucs_lightning
├──conf
│     ├─train_test_config.yaml
│     ├─infer_config.yaml
│     │
│
├──demucs
│     ├─demucs.py
│     ├─hdemucs.py
│     ├─other custome modules
│
├──requirements.txt
├──train.py
├──test.py
├──inference.py
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

## Logging

TensorBoard logging is used by default. If you want to use the
WandbLogger instead (recommended!), either edit `logger` in
`conf/train_test_config.yaml` or postpend `logger=wandb` to all
your commands.

## Training
If it is your first time running the repo, you can use the argument `download=True` to automatically download and setup the `musdb18hq` dataset. Otherwise, you can omit this  argument.

### Demucs
It requires `16,885 MB` of GPU memory. If you do not have enough GPU memory, please read [this section](#Training-with-a-less-powerful-GPU).

```bash
python train.py gpus=[0] model=Demucs download=True
```

### HDemucs
It requires `19,199 MB` of GPU memory.
```bash
python train.py gpus=[0] model=HDemucs download=True
```

## Resume from Checkpoints
It is possible to continue training from an existing checkpoint by passing the `resume_checkpoint` argument. By default, hydra saves all the checkpoints at `'outputs/YYYY-MM-DD/HH-MM-SS/XXX_experiment_epoch=XXX_augmentation=XXX/version_1/checkpoints/XXX.ckpt'`. For example, if you have a checkpoint trained with 32-bit precision for 100 epochs already via the following command:

```bash
python train.py gpus=[0] trainer.precision=32 epochs=100
```

And now you want to train for 50 epochs more, then you can use the following CLI command:

```bash
python train.py gpus=[0] trainer.precision=16 epochs=150 resume_checkpoint='outputs/2022-05-24/21-20-17/Demucs_experiment_epoch=360_augmentation=True/version_1/checkpoints/e=123-TRAIN_loss=0.08.ckpt'
```

You can always move you checkpoints to a better place to shorten the path name.

## Training with a less powerful GPU
It is possible to reduce the GPU memory required to train the models by using the following tricks. But it might affect the model performance.
### Reduce Batch Size
You can reduce the batch size to `2`. By doing so, it only requires `10,851 MB` of GPU memory.
```bash
python train.py batch_size=2 augment.remix.group_size=2 model=Demucs
```

### Disable Augmentation 
You can futher reduce the batch size to `1` if data augmentation is disabled. By doing so, it only requires `7,703 MB` of GPU memory.
```bash
python train.py batch_size=1 data_augmentation=False model=Demucs
```


### Reduce Audio Segment Length
You can reduce the audio segment length to only `6`. By doing so, it only requires `6,175 MB` of GPU memory.
```bash
python train.py batch_size=1 data_augmentation=False segment=6 model=Demucs
```

## Testing pretrained model
You can use `test.py` to evaluate the pretrained model directly by using an existing checkpoint. You can give the checkpoint path via `resume_checkpoint` argument.

```bash
python test.py resume_checkpoint='outputs/2022-05-24/21-20-17/Demucs_experiment_epoch=360_augmentation=True/version_1/checkpoints/e=123-TRAIN_loss=0.08.ckpt'
```

## Half-Precision Training
By default, pytorch lightning uses 32-bit precision for training. To use 16-bit precision (half-precision), you can specify `trainer.precision`:

```bash
python train.py trainer.precision=16
```

Double-precision is also supported by specifying `trainer.precision=64`.



## Default settings 
The full list of arguments and their default values can be found in `conf/config.yaml`.

__gpus__: Select which GPU to use. If you have multiple GPUs on your machine and you want to use GPU:2, you can set `gpus=[2]`. If you want to use DDP (multi-GPU training), you can set `gpus=2`, it will automatically use the first two GPUs avaliable in your machine. If you want to use GPU:0, GPU:2, and GPU:3 for training, you can set `gpus=[0,2,3]`.

__download__: When set to `True`, it will automatically download and setup the dataset. Default as `False`

__data_root__: Select the location of your dataset. If `download=True`, it will become the directory that the dataset is going to be downloaded to. Default as `'./musdb18hq'`

__model__: Select which version of demucs to use. Default model of this repo is Hybrid Demucs (v3). You can switch to Demucs (v2) by setting the `model=Demucs`.

__samplerate__: The sampling rate for the audio. Default as `44100`.

__epochs__: The number of epochs to train the model. Default as `360`.

__optim.lr__: Learning rate of the optimizer. Default as `3e-4`.


## Inferencing
You are able to apply your trained model weight on your own audio file by using `inference.py`. Some nesscesary argument are the following:

* `checkpoint` refers to the path of trained model weight checkpoint file
* `infer_audio_folder_path` refers to the path of your audio folder where has all the audios inside
* `infer_audio_ext` refer to the type of your audio. Default value is `'wav'`

```bash
python inference.py infer_audio_folder_path='../../infer_audio' checkpoint='outputs/2022-05-24/21-20-17/Demucs_experiment_epoch=360_augmentation=True/version_1/checkpoints/e=123-TRAIN_loss=0.08.ckpt'
```

By default, hydra saves all the seperated audio in the `outputs` folder.
## Development

If you are a developer on this repo, please run:

```
pre-commit install
```

