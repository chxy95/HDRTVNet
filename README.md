# HDRTVNet [[Paper Link]](https://arxiv.org/abs/2108.07978)

### A New Journey from SDRTV to HDRTV
By Xiangyu Chen*, Zhengwen Zhang*, [Jimmy S. Ren](https://scholar.google.com.hk/citations?hl=zh-CN&user=WKO_1VYAAAAJ), Lynhoo Tian, [Yu Qiao](https://scholar.google.com/citations?user=gFtI-8QAAAAJ&hl=zh-CN) and [Chao Dong](https://scholar.google.com.hk/citations?user=OSDCB0UAAAAJ&hl=zh-CN)

(* indicates equal contribution)

## Still Updating （Aug. 23th, 2021）...

## Overview
Simplified SDRTV/HDRTV formation pipeline:
<img src="https://raw.githubusercontent.com/chxy95/HDRTVNet/master/images/Formation_Pipeline.png" width="600"/>

Overview of the method:
<img src="https://raw.githubusercontent.com/chxy95/HDRTVNet/master/images/Network_Structure.png" width="600"/>

## Getting Started

1. [Dataset](#dataset)
2. [Configuration](#configuration)
3. [How to test](#how-to-test)
4. [How to train](#how-to-train)
5. [Metrics](#metrics)
6. [Visualization](#visualization)

### Dataset
We conduct a dataset using videos with 4K resolutions under HDR10 standard and their counterpart SDR versions from Youtube. The dataset consists of a training set with 1235 image pairs and a test set with 117 image pairs. Please refer to the paper for the details on the processing of the dataset. The dataset can be downloaded from [Baidu Netdisk](https://pan.baidu.com/s/1TwXnBzeV6TlD3zPvuEo8IQ) (code: 6qvu) or [OneDrive](https://uofmacau-my.sharepoint.com/:f:/g/personal/yc17494_umac_mo/Ep6XPVP9XX9HrZDUR9SmjdkB-t1NSAddMIoX3iJmGwqW-Q?e=dNODeW) (code: HDRTVNet).

We also provide the original Youtube links of these videos, which can be found in this [file](https://raw.githubusercontent.com/chxy95/HDRTVNet/master/links.txt). Note that we cannot provide the download links since we do not have the copyright to distribute. **Please download this dataset only for academic use.**

### Configuration

Please refer to the [requirements](https://raw.githubusercontent.com/chxy95/HDRTVNet/master/requirements.txt).

### How to test

We provide the pretrained models to test, which can be downloaded from [Baidu Netdisk](https://pan.baidu.com/s/1OSLVoBioyen-zjvLmhbe2g) (code: 2me9) or [OneDrive](https://uofmacau-my.sharepoint.com/:f:/g/personal/yc17494_umac_mo/EteMb8FVYE5GqILE2mV-1W8B0-S_ynjt2gAgHkDH9LgkMg?e=EnBn3Q) (code: HDRTVNet). Since our method is casaded of three steps, the results also need to be inferenced step by step. 

- For the first part of AGCM, make sure the paths of `dataroot_LQ`, `dataroot_GT` and `pretrain_model_G` in `./codes/options/test/test_AGCM.yml` are correct, then run
```
cd codes
python test.py -opt options/test/test_AGCM.yml
```
The test results will be saved to `./results/Adaptive_Global_Color_Mapping`.

- For the second part of LE, modify the `dataroot_LQ` into 
- For the last part of HG,
### How to train

### Metrics

### Visualization

## Citation
If our work is helpful to you, please cite our paper:

    @inproceedings{chen2021new,
      title={A New Journey from SDRTV to HDRTV}, 
      author={Chen, Xiangyu and Zhang, Zhengwen and Ren, Jimmy S. and Tian, Lynhoo and Qiao, Yu and Dong, Chao},
      booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
      year={2021}
    }
