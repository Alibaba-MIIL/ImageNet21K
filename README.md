# ImageNet-21K Pretraining for the Masses

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/imagenet-21k-pretraining-for-the-masses/multi-label-classification-on-ms-coco)](https://paperswithcode.com/sota/multi-label-classification-on-ms-coco?p=imagenet-21k-pretraining-for-the-masses)
<br>
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/imagenet-21k-pretraining-for-the-masses/multi-label-classification-on-pascal-voc-2007)](https://paperswithcode.com/sota/multi-label-classification-on-pascal-voc-2007?p=imagenet-21k-pretraining-for-the-masses)
<br>
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/imagenet-21k-pretraining-for-the-masses/fine-grained-image-classification-on-stanford)](https://paperswithcode.com/sota/fine-grained-image-classification-on-stanford?p=imagenet-21k-pretraining-for-the-masses)
<br>
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/imagenet-21k-pretraining-for-the-masses/image-classification-on-cifar-100)](https://paperswithcode.com/sota/image-classification-on-cifar-100?p=imagenet-21k-pretraining-for-the-masses)

<br>[Paper](https://arxiv.org/pdf/2104.10972v4.pdf) |
[Pretrained models](MODEL_ZOO.md)

Official PyTorch Implementation

> Tal Ridnik, Emanuel Ben-Baruch, Asaf Noy, Lihi Zelnik-Manor<br/> DAMO Academy, Alibaba
> Group

**Abstract**

ImageNet-1K serves as the primary dataset for pretraining deep learning models for computer vision tasks. ImageNet-21K dataset, which contains more pictures and classes, is used less frequently for pretraining, mainly due to its complexity, and underestimation of its added value compared to standard ImageNet-1K pretraining.
This paper aims to close this gap, and make high-quality efficient pretraining on ImageNet-21K available for everyone.
Via a dedicated preprocessing stage, utilizing WordNet hierarchies, and a novel training scheme called semantic softmax, we show that different models, including small mobile-oriented models, significantly benefit from ImageNet-21K pretraining on numerous datasets and tasks.
We also show that we outperform previous ImageNet-21K pretraining schemes for prominent new models like ViT.
Our proposed pretraining pipeline is efficient, accessible, and leads to SoTA reproducible results, from a publicly available dataset.

<br>
<br>
<img src="./pics/pic1.png" align="center" width="450" height="350"></td>

## 24/10/2021 Update - New article released for partially annotated datasets
Checkout our new project, [Multi-label Classification with Partial Annotations using Class-aware Selective Loss](https://github.com/Alibaba-MIIL/PartialLabelingCSL), where we presnet a top solution
to multi-label datasets which are using partial annotation (such as Open Images and LVIS). Our soluion off-course uses large-scale TResNet pretraining.


<!--## 05/08/2021 Update - Training From Random Initilization Script Released-->
<!--We release the following [script](./train_single_label_from_scratch.py), that reproduces the article results with random initialization, instead of initalizing from ImageNet-1K. The only change is that 140 epochs of training are needed, instead of 80, and we use SGD optimizer with a higher learning rate.-->
<!--We still recommend initializing your models from standard ImageNet-1K pretraining to reduce training times and save rainforests.-->

<!--Note that we also released a new version of the article in arxiv, with additional interesting ablation tests. For example - comparison to Open Images pretraining, and more-->
<!--comparisons on non-classification tasks. Thanks to Matan Karklinsky for his help with Inria-holidays dataset.-->


## 01/08/2021 Update - "ImageNet-21K Pretraining for the Masses" Was Accepted to NeurIPS 2021 !
We are very happy to announce that "ImageNet-21K Pretraining for the Masses" article was accepted to NeurIPS 2021 (Datasets and Benchmarks [Track](https://nips.cc/Conferences/2021/CallForDatasetsBenchmarks)). OpenReview is
available [here](https://openreview.net/forum?id=Zkj_VcZ6ol&noteId=1oUacUMpIbg).

<!--## 17/05/2021 Update - ImageNet-21K-P is Now Available for Downloading From the Official ImageNet Site-->
<!--ImageNet-21K-P processed dataset, based on ImageNet-21K winter release, is now available-->
<!--for easy downloading via the offical [ImageNet site](https://image-net.org/download-images.php).-->
<!--See more details on the different versions of ImageNet-21K-P in [here](./dataset_preprocessing/processing_instructions.md).-->


## Getting Started

### (0) Visualization and Inference Script
First you can play and do inference on dedicated images using the following [script](./visualize_detector.py).
An example result:

<img src="./pics/dog_inference.png" align="center" width="500" >

### (1) Pretrained Models  on ImageNet-21K-P Dataset
| Backbone  |  ImageNet-21K-P semantic<br> top-1 Accuracy <br>[%] | ImageNet-1K<br> top-1 Accuracy <br>[%] | Maximal <br> batch size | Maximal <br> training speed <br>(img/sec) | Maximal <br> inference speed <br>(img/sec) |
| :------------: | :--------------: | :--------------: | :--------------: | :--------------: | :--------------: |
[MobilenetV3_large_100](https://miil-public-eu.oss-eu-central-1.aliyuncs.com/model-zoo/ImageNet_21K_P/models/mobilenetv3_large_100_miil_21k.pth) | 73.1 | 78.0 | 488 | 1210 | 5980 |
[OFA_flops_595m_s](https://miil-public-eu.oss-eu-central-1.aliyuncs.com/model-zoo/ImageNet_21K_P/models/ofa_flops_595m_s_miil_21k.pth) | 75.0 | 81.0 | 288 | 500 | 3240 |
[ResNet50](https://miil-public-eu.oss-eu-central-1.aliyuncs.com/model-zoo/ImageNet_21K_P/models/resnet50_miil_21k.pth) | 75.6 | 82.0 | 320 | 720 | 2760 |
[TResNet-M](https://miil-public-eu.oss-eu-central-1.aliyuncs.com/model-zoo/ImageNet_21K_P/models/tresnet_m_miil_21k.pth) | 76.4 | 83.1 | 520 | 670 | 2970 |
[TResNet-L (V2)](https://miil-public-eu.oss-eu-central-1.aliyuncs.com/model-zoo/ImageNet_21K_P/models/tresnet_l_v2_miil_21k.pth) | 76.7 | 83.9 | 240 | 300 | 1460 |
[ViT-B-16](https://miil-public-eu.oss-eu-central-1.aliyuncs.com/model-zoo/ImageNet_21K_P/models/vit_base_patch16_224_miil_21k.pth) | 77.6 | 84.4 | 160 | 340 | 1140 |

See [here](MODEL_ZOO.md) for more details.
<br>
We highly recommend to start working with ImageNet-21K by testing these weights against standard ImageNet-1K pretraining, and comparing results on your relevant downstream tasks.
After you will see a significant improvement, proceed to pretraining new models.

Note that some of our models, with 21K and 1K pretraining, are also avaialbe via the excellent [timm](https://github.com/rwightman/pytorch-image-models) package:
```
21K:
model = timm.create_model('mobilenetv3_large_100_miil_in21k', pretrained=True)
model = timm.create_model('tresnet_m_miil_in21k', pretrained=True)
model = timm.create_model('vit_base_patch16_224_miil_in21k', pretrained=True)
model = timm.create_model('mixer_b16_224_miil_in21k', pretrained=True)


1K:
model = timm.create_model('mobilenetv3_large_100_miil', pretrained=True)
model = timm.create_model('tresnet_m', pretrained=True)
model = timm.create_model('vit_base_patch16_224_miil', pretrained=True)
model = timm.create_model('mixer_b16_224_miil', pretrained=True)
```
Using this [link](https://github.com/rwightman/pytorch-image-models/blob/master/results/results-imagenet.csv) you can make sure we indeed reach the reported accuracies in the article.

### (2) Obtaining and Processing the Dataset
See instructions for obtaining and processing the dataset in [here](./dataset_preprocessing/processing_instructions.md).


### (3) Training Code
To use the training code, first download the relevant ImageNet-21K-P semantic tree file to your local ./resources/ folder:
- fall11 [semantic tree](https://miil-public-eu.oss-eu-central-1.aliyuncs.com/model-zoo/ImageNet_21K_P/resources/fall11/imagenet21k_miil_tree.pth)
- winter21 [semantic tree](https://miil-public-eu.oss-eu-central-1.aliyuncs.com/model-zoo/ImageNet_21K_P/resources/winter21/imagenet21k_miil_tree.pth)



Example of semantic softmax training:
```
python train_semantic_softmax.py \
--batch_size=4 \
--data_path=/mnt/datasets/21k \
--model_name=mobilenetv3_large_100 \
--model_path=/mnt/models/mobilenetv3_large_100.pth \
--epochs=80
```
For shortening the training, we initialize the weights from standard ImageNet-1K. Recommended to use ImageNet-1K weights from timm [repo](https://github.com/rwightman/pytorch-image-models).

### (4) Transfer Learning Code
See [here](Transfer_learning.md) for reproduction code, that show how miil pretraining not only improves transfer learning results, but also makes MLP models more stable and robust for hyper-parameters selection.

## Additional SoTA results
The results in the article are comparative results, with fixed hyper-parameters.
In addition, using our pretrained models, and a dedicated training scheme with adjusted hyper-parameters per dataset (resolution, optimizer, learning rate), we were able
to achieve SoTA results on several computer vision dataset - MS-COCO, Pascal-VOC, Stanford Cars and CIFAR-100.

We will share our models' checkpoints to validate our scores.

<!--## To be added-->
<!--- KD training code-->
<!--- Inference code-->
<!--- Model weights after transferred to ImageNet-1K-->
<!--- Downstream training code.-->
<!--- More-->

## Citation
```
@misc{ridnik2021imagenet21k,
      title={ImageNet-21K Pretraining for the Masses}, 
      author={Tal Ridnik and Emanuel Ben-Baruch and Asaf Noy and Lihi Zelnik-Manor},
      year={2021},
      eprint={2104.10972},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
