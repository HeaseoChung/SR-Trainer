<a href="https://github.com/HeaseoChung/DL-Optimization/tree/master/Python/TensorRT/x86"><img src="https://img.shields.io/badge/-Documentation-brightgreen"/></a>

# SR Trainer - mlflow
- SR trainer optimized with mlflow 👏👏

## Contents
- [Updates](#updates)
- [Models](#models)
- [Features](#features)
- [Usage](#usage)
- [Citation](#citation)

## Updates
- **_News (2024-03-08)_**: Fixed minor issues and being preparing to implement MLFlow
- **_News (2023-12-08)_**: Implemented warm start and finetune strategy for training
- **_News (2023-08-21)_**: Implemented profiling model to check its FLOPs, inference time, memory footprint, etc
- **_News (2023-05-12)_**: Implemented conversion for ONNX and TensorRT inference
- **_News (2023-05-01)_**: Refactored codes
- **_News (2023-02-13)_**: Implemented profiler to check spec of models
- **_News (2023-01-10)_**: Implemented Wandb to monitor status of training
- **_News (2023-01-01)_**: Implemented Knowledge Distillation
- **_News (2022-12-30)_**: Refactoring codes for tester and metricer
- **_News (2022-12-30)_**: Refactored codes for train
- **_News (2022-12-22)_**: Implemented interlace degradation
- **_News (2022-08-25)_**: Implemented calculator for quantitative score
- **_News (2022-07-23)_**: Implemented multi-gpus train approach using pytorch distributed data parallel
- **_News (2022-07-05)_**: Implemented test codes for video and image
- **_News (2022-07-02)_**: Implemented train codes for super-resolution models such as EDSR, BSRGAN, RealESRGAN, SwinIR and SCUNet

## Models
- BSRGAN
- EDSR
- Real-ESRGAN
- SCUNET
- SwinIR

## Features
- Training super-resolution using NET, GAN, Knowledge Distillation approaches
- Testing super-resolution models for image and video
- Converting PyTorch based super-resolution models to ONNX

## Train Approaches
- NET
- GAN
- WS
- FT

## SR Trainer Structures

### 1. Configs to manage hyperparameters for train
```
configs/
├── models
│   ├── BSRGAN.yaml
│   ├── EDSR.yaml
│   ├── RealESRGAN.yaml
│   └── SCUNET.yaml
│   └── SwinIR.yaml
├── test
│   ├── image.yaml
│   └── video.yaml
├── train
│   ├── GAN.yaml
│   └── NET.yaml
│   └── WS.yaml
│   └── KD.yaml
├── test.yaml
└── train.yaml
```

### 2. Scripts for train and test
```
├── archs
│   ├── BSRGAN
│   │   ├── README.md
│   │   └── models.py
│   ├── EDSR
│   │   ├── README.md
│   │   └── models.py
│   ├── RealESRGAN
│   │   ├── README.md
│   │   └── models.py
│   ├── SCUNet
│   │   ├── README.md
│   │   └── models.py
│   ├── SwinIR
│   │   └── models.py
│   ├── Utils
│   │   ├── __init__.py
│   │   ├── blocks.py
│   │   └── utils.py
│   ├── __init__.py
├── data
│   ├── __init__.py
│   ├── dataset.py
│   └── degradation.py
├── learning_rate_scheduler
│   ├── __init__.py
├── loss
│   ├── __init__.py
│   ├── gan_loss.py
│   ├── l1_charbonnier_loss.py
│   └── perceptual_loss.py
├── metric
│   └── metrics.py
├── metricer.py
├── models.py
├── optim
│   ├── __init__.py
├── tester.py
├── trainer.py
└── utils.py
```


## How to use SR Trainer

### 1. Modifying train.yaml in the configs directory with your own hyperparamters to train

#### Train

```yaml
### train.yaml 
### A user should change model_name and train_type to train

hydra:
  run:
    dir: ./outputs/train/${now:%Y-%m-%d}/${now:%H-%M-%S}

defaults:
  - _self_
  - train: train_type # PSNR, GAN
  - models: model_name # EDSR
```

```yaml
### PSNR or GAN.yaml

### A user should specify train directory in datasets
dataset:
  train:
    type: ImagePairDegradationDataset
    hr_dir: datasets_path
```

```yaml
### model.yaml
### A user should specify the path of checkpoint in order to resume your train
generator:
  path: "" # /model_zoo/edsr.pth

discriminator:
  path: "" # /model_zoo/discriminator.pth
```

#### Test

```yaml
### test.yaml 
### A user should change model_name and test_type to test with various models

hydra:
  run:
    dir: ./outputs/test/${now:%Y-%m-%d}/${now:%H-%M-%S}

defaults:
  - _self_
  - test: test_type # image, video
  - models: model_name # EDSR
```

```yaml
### model.yaml
### A user should specify the path of pre-trained to load weights, in order to inference your model
generator:
  path: "" # /model_zoo/edsr.pth
```

#### Valid

```yaml
### valid.yaml 
### A user should change model_name and vaild_type to valid with various models

hydra:
  run:
    dir: ./outputs/SCUNET/valid/${now:%Y-%m-%d}/${now:%H-%M-%S}

defaults:
  - _self_
  - valid: valid_typr # quantitative
  - models: model_name # EDSR
```

```yaml
### quantitative.yaml
### A user should specify the path of hr_dir & lr_dir to load custom dataset and metrics to valid your models
save_path: "quantitative"
dataset:
  hr_dir: "/workspace/SuperResolution/example/Ref"
  lr_dir: "/workspace/SuperResolution/example/Dis"

metrics: [psnr, ssim, lpips, erqa]
```

#### Profiling

```yaml
hydra:
  run:
    dir: ./outputs/EDSR/profile/${now:%Y-%m-%d}/${now:%H-%M-%S}

defaults:
  - _self_
  - models: EDSR

data:
  batch: 1
  channel: 3
  width: 1920
  height: 1088
```

#### ONNX Converting
```yaml
hydra:
  run:
    dir: ./outputs/EDSR/onnx/${now:%Y-%m-%d}/${now:%H-%M-%S}

defaults:
  - _self_
  - models: EDSR

data:
  dynamic_shape: True
  batch: 1
  channel: 3
  width: 1920
  height: 1080
```

### 3. Run

```python
### trainer.py for train a model using single GPU
CUDA_VISIBLE_DEVICES=0 python trainer.py

### trainer.py for train a model using multiple GPUs
CUDA_VISIBLE_DEVICES=0,1,2,3 python trainer.py
```

```python
### tester.py for test a model
CUDA_VISIBLE_DEVICES=0 python tester.py
```

```python
### pytorch2onnx.py for converting a model to ONNX
CUDA_VISIBLE_DEVICES=0 python pytorch2onnx.py
```

```python
### profiler.py for profiler a model
CUDA_VISIBLE_DEVICES=0 python profiler.py
```

## Citation
If the repository helps your research or work, please consider citing AutomaticSR.<br>
The following is a BibTeX reference. The BibTeX entry requires the `url` LaTeX package.

``` latex
@misc{chung2022AutomaticSR,
  author =       {Heaseo Chung},
  title =        {{AutomaticSR}: Open source for various super-resolution trainer and tester},
  howpublished = {\url{https://github.com/HeaseoChung/Super-resolution}},
  year =         {2022}
}
```