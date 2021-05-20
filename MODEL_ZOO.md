#  Pretrained models

We provide a collection of models trained with semantic softmax on ImageNet-21K-P dataset. All results are on input resolution of 224.
<br>For proper comparison between the models, we also provide some throughput metrics.


| Backbone  |  ImageNet-21K-P semantic<br> top-1 Accuracy <br>[%] | ImageNet-1K<br> top-1 Accuracy <br>[%] | Maximal <br> batch size | Maximal <br> training speed <br>(img/sec) | Maximal <br> inference speed <br>(img/sec) |
| :------------: | :--------------: | :--------------: | :--------------: | :--------------: | :--------------: |
[MobilenetV3_large_100](https://miil-public-eu.oss-eu-central-1.aliyuncs.com/model-zoo/ImageNet_21K_P/models/mobilenetv3_large_100_miil_21k.pth) | 73.1 | 78.0 | 488 | 1210 | 5980 |
[OFA_flops_595m_s](https://miil-public-eu.oss-eu-central-1.aliyuncs.com/model-zoo/ImageNet_21K_P/models/ofa_flops_595m_s_miil_21k.pth) | 75.0 | 81.0 | 288 | 500 | 3240 |
[ResNet50](https://miil-public-eu.oss-eu-central-1.aliyuncs.com/model-zoo/ImageNet_21K_P/models/resnet50_miil_21k.pth) | 75.6 | 82.0 | 320 | 720 | 2760 |
[TResNet-M](https://miil-public-eu.oss-eu-central-1.aliyuncs.com/model-zoo/ImageNet_21K_P/models/tresnet_m_miil_21k.pth) | 76.4 | 83.1 | 520 | 670 | 2970 |
[TResNet-L (V2)](https://miil-public-eu.oss-eu-central-1.aliyuncs.com/model-zoo/ImageNet_21K_P/models/tresnet_l_v2_miil_21k.pth) | 76.7 | 83.9 | 240 | 300 | 1460 |
[ViT_B_16_patch_224](https://miil-public-eu.oss-eu-central-1.aliyuncs.com/model-zoo/ImageNet_21K_P/models/vit_base_patch16_224_miil_21k.pth) | 77.6 | 84.4 | 160 | 340 | 1140 |

To initialize the different models and properly load the weights, use this [file](./src_files/models/utils/factory.py).

use the following models names (--model_name):
tresnet_m, tresnet_l, ofa_flops_595m_s, resnet50, vit_base_patch16_224, mobilenetv3_large_100

Notes
- Maximal training and inference speeds were calculated on NVIDIA V100 GPU, with 90% of maximal batch size.
- ViT model highly benefits from O2 mixed-precision training and inference.  O1 mixed-precision speeds (torch.autocast) are lower.
- We are still optimising ViT hyper parameters on ImageNet-1K training. Accuracy would probably be higher in the future.
- Our ofa_flops_595m model is slightly different than the orignal model - we converted all hard-sigmoids to regular sigmoids, since they are faster, both on CPU and GPU, and gives better scores. Hence we renamed the model to 'ofa_flops_595m_s'.
