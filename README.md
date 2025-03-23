# SeqCSIST: Sequential Closely-Spaced Infrared Small Target Unmixing

## Introduction
This repository contains the official implementation of our paper **"SeqCSIST: Sequential Closely-Spaced Infrared Small Target Unmixing"**. Our work introduces:
- **A novel task**: CSIST Unmixing, which aims to detect [all targets in the form of sub-pixel localization from a highly dense CSIST group].
- **A new dataset**: SeqCSIST, specifically designed for [multi-frame CSIST Umixing].
- **An End-to-End Framework**: Our approach outperforms existing methods by [].

## Dataset
- **Number of samples**: [100,000 frames organized into 5,000 random trajectories]
- **Download**: []

## Model
Our model consists of three main modules:
- **[sparsity-driven feature extraction module]**: []
- **[positional encoding module]**: []
- **[emporal Deformable Feature Alignment (TDFA) module]**: []

### Architecture
![Model Architecture]()

## Installation
To set up the environment, run:
```bash
conda env create -f environment.yml
conda activate speed
mim install mmcv==2.0.1
```

## Training
To train the model, run:
```bash

```

## Evaluation
To evaluate on the test set, run:
```bash

```

## Results
Our method achieves state-of-the-art performance on SeqCSIST Task:


## Citation
If you find this work useful, please cite our paper:
```
@article{your_paper,
  title={SeqCSIST: Sequential Closely-Spaced Infrared Small Target Unmixing},
  author={Ximeng Zhai, Bohan Xu, Yaohong Chen, Hao Wang, Kehua Guo, Yimian Dai},
  journal={ArXiv/IEEE Transactions on Geoscience and Remote Sensing},
  year={2025}
}
```

