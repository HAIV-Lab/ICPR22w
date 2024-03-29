# Lightweight Human Pose Estimation Using Loss Weighted by Target Heatmap

This is an official implementation of [*Lightweight Human Pose Estimation Using Loss Weighted by Target Heatmap*](https://link.springer.com/chapter/10.1007/978-3-031-37660-3_5) that is accepted to ICPR TCAP workshop and honored as Best Paper Award (Honorable Mention). The code is developed based on the repository of [MMPose](https://github.com/open-mmlab/mmpose).

## Introduction

In this work, we lighten the computation cost and parameters of the deconvolution head network in SimpleBaseline and introduce an attention mechanism that utilizes original, inter-level, and intra-level information to intensify the accuracy. Additionally, we propose a novel loss function called heatmap weighting loss, which generates weights for each pixel on the heatmap that makes the model more focused on keypoints. Experiments demonstrate our method achieves a balance between performance, resource volume, and inference speed. 

## Results

### results on COCO val2017

| Input Size | #Params | GFLOPs |  AP  | AP<sup>50</sup> |  AP<sup>75</sup>  |  AP<sup>M</sup>  |  AP<sup>L</sup>  |  AR  |
| :--------: | :-----: | :----: | :--: | :--: | :--: | :--: | :--: | :--: |
| 256 × 192  |  3.1M   |  0.58  | 65.8 | 87.7 | 74.1 | 62.6 | 72.4 | 72.1 |
| 384 × 288  |  3.1M   |  1.30  | 69.9 | 88.8 | 77.5 | 66.0 | 76.7 | 75.5 |

### Inference Speed

Inference speed on Intel Core i7-10750H CPU and NVIDIA GTX 1650Ti (Notebooks) GPU.

|    Model     |  AP  | FPS(GPU) | FPS(CPU) | GFLOPs |
| :----------: | :--: | :------: | :------: | :----: |
| MobileNetV2  | 64.6 |   57.6   |   19.3   |  1.59  |
| ShuffleNetV2 | 59.9 |   51.8   |   20.2   |  1.37  |
|    ViPNAS    | 67.8 |   22.6   |   4.6    |  0.69  |
| Lite-HRNet18 | 64.8 |   12.8   |   10.3   |  0.20  |
| Lite-HRNet30 | 67.2 |   7.5    |   6.2    |  0.31  |
|     Ours     | 65.8 |   55.2   |   18.3   |  0.58  |

<img src="/imgs/speed.png" alt="speed" style="zoom:40%;" />

## Installation and Preparation

Please refer to MMPose's [documentation](https://mmpose.readthedocs.io/en/latest/) and add the file to corresponding direction.

## Environment

The code is developed using python 3.7 on Ubuntu 20.04. The code is developed and tested using a single NVIDIA RTX 3090 GPU. Other platforms or GPU cards are not fully tested.

## Cite us
```
@inproceedings{li2022lightweight,
  title={Lightweight Human Pose Estimation Using Loss Weighted by Target Heatmap},
  author={Li, Shiqi and Xiang, Xiang},
  booktitle={International Conference on Pattern Recognition},
  pages={64--78},
  year={2022},
  organization={Springer}
}
```

## Acknowledgement

Thanks to:

- [MMPose](https://github.com/open-mmlab/mmpose)
- [SimpleBaseline](https://github.com/microsoft/human-pose-estimation.pytorch)
