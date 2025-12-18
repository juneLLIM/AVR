# Code for Acoustic Field Reconstruction 
### [NeurIPS'24 Spotlight] AVR: Acoustic Volume Rendering for Neural Impulse Response Fields
### [NeurIPS'25 Paper] Versa: Resounding Acoustic Fields with Reciprocity

[[AVR arXiv](https://arxiv.org/abs/2411.06307)] [[AVR Website](https://zitonglan.github.io/project/avr/avr.html)] [[AcoustiX Code](https://github.com/penn-waves-lab/AcoustiX)] [[BibTex](#citation)] 

[[Versa arXiv](https://arxiv.org/abs/2510.20602)] [[Versa Website](https://zitonglan.github.io/project/versa/versa.html)]

![Figure Description](assets/method.png)

## Contents
This repo contains the official implementation for AVR. For our simulator: [AcoustiX Code](https://github.com/penn-waves-lab/AcoustiX), please use another repo.
- [Abstract](#abstract)
- [Codebase](#codebase)
- [Usage](#usage)
- [Citation](#citation)


## Updates
- 2025.12.18: Add Versa repo.

<!-- ## Abstract
Realistic audio synthesis that captures accurate acoustic phenomena is essential for creating immersive experiences in virtual and augmented reality. Synthesizing the sound received at any position relies on the estimation of impulse response (IR), which characterizes how sound propagates in one scene along different paths before arriving at the listener’s position. In this paper, we present Acoustic Volume Rendering (AVR), a novel approach that adapts volume rendering techniques to model acoustic impulse responses. While volume rendering has been successful in modeling radiance fields for images and neural scene representations, IRs present unique challenges as time-series signals. To address these challenges, we introduce frequency-domain volume rendering and use spherical integration to fit the IR measurements. Our method constructs an impulse response field that inherently encodes wave propagation principles and achieves state-of-the-art performance in synthesizing impulse responses for novel poses. Experiments show that AVR surpasses current leading methods by a substantial margin. Additionally, we develop an acoustic simulation platform, AcoustiX, which provides more accurate and realistic IR simulations than existing simulators. -->

## Codebase
Below is the instructions on how to install and set up the project.

### Requirements (in addition to the common python stack)

* pytorch
* numpy
* scipy
* matplotlib
* librosa
* auraloss

In addition to above common python packages, we also use [tinycudann](https://github.com/NVlabs/tiny-cuda-nn) to speed up the ray sampling. Installation of this repo and python extension is shown below.  

```sh
# install tiny-cuda-nn PyTorch extension
pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
```

### Project structure

```sh
AVR/
├── config_files/          # training and testing config files
│   ├── avr_raf_furnished.yml
│   ├── avr_raf_empty.yml
│   ├── avr_meshrir.yml
│   └── avr_simu.yml       
├── logs/                  # Log files
│   ├── meshrir            # Meshrir logs
│   ├── RAF                # RAF logs
│   └── simu               # Simulation logs
├── tensorboard_logs/      # TensorBoard log files
├── data/                  # Dataset 
├── utils/                 # Utility scripts
│   ├── criterion.py       # Loss functions   
│   ├── logger.py          
│   ├── spatialization.py  # Audio spatialization        
│   └── metric.py          # Metrics calculation.
├── tools/                 # Tools to create datasets and more   
│   └── meshrir_split.py   # Create meshrir dataset split
├── avr_runner.py          # AVR runner
├── datasets_loader.py     # dataloader for different datasets
├── model.py               # network
├── renderer.py            # acoustic rendering file
├── README.md              # Project documentation
└── .gitignore             # Git ignore file
```

## Example Usage

* Train AVR on *RAF-Furnished* dataset
```sh
python avr_runner.py --config ./config_files/avr_raf_furnished.yml --dataset_dir ./data/RAF/FurnishedRoomSplit
```

* Train AVR on *RAF-Empty* dataset
```sh
python avr_runner.py --config ./config_files/avr_raf_empty.yml --dataset_dir ./data/RAF/EmptyRoomSplit
```

* Train AVR on *MeshRIR* dataset
```sh
python avr_runner.py --config ./config_files/avr_meshrir.yml  --dataset_dir ./data/MeshRIR
```

## Create your own dataset
MeshRIR dataset: Refer to [Create Meshrir Dataset Instructions](tools/README.md##Instructions-on-creating-Mesh-RIR-S1-M3969-split)

## Results and Visualizations
We show some visualization results from our paper:

We show the impusle response frequency spatial distribution with ground truth and different baseline methods. This is a bird-eye view of signal distributions.
![Figure Description](assets/wave_distributions.png)

We show the estimated impulse response from different datasets and methods.
![Figure Description](assets/signal.png)


## Citation
If you find this project to be useful for your research, please consider citing the paper.
```
@inproceedings{lanresounding,
  title={Resounding Acoustic Fields with Reciprocity},
  author={Lan, Zitong and Hao, Yiduo and Zhao, Mingmin},
  booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems}
}

@inproceedings{lanacoustic,
  title={Acoustic Volume Rendering for Neural Impulse Response Fields},
  author={Lan, Zitong and Zheng, Chenhao and Zheng, Zhiwei and Zhao, Mingmin},
  booktitle={The Thirty-eighth Annual Conference on Neural Information Processing Systems}
}
```
