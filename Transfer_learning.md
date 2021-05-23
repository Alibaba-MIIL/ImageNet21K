#  Transfer Learning Code

Due to commercial limitations, we cannot share at this time the transfer learning code used in the article.
However, using the excellent [timm](https://github.com/rwightman/pytorch-image-models) package, the article result can be reproduced almost completely.
Specifically, timm package enables to compare official pretraining and miil pretraining of ViT and Mixer model, and validate the improvement in
transfer learning results. This comparison also enables to show how miil pretraining stabilizes transfer learning results, and make them far less susceptible to hyper-parameter selection.

An example training code on cifar100:
```
python train.py \
/Cifar100Folder/ \
-b=128 \
--img-size=224 \
--epochs=50 \
--color-jitter=0 \
--squish \
--amp \
--sched='cosine' \
--model-ema --model-ema-decay=0.995 --squish --reprob=0.5 --smoothing=0.1 \
--nonstrict_checkpoint --min-lr=1e-8 --warmup-epochs=3 --train-interpolation=bilinear --aa=v0 \
--pretrained \
--lr=2e-4 \
--model=mixer_b16_224_in21k \
--opt=adam --weight-decay=1e-4 \
```

These are the result we got for the official 21k pretrain (--model=mixer_b16_224_in21k) and miil 21k pretrain (--model=mixer_b16_224_miil_in21k), for different hyper-parameter selection:

<!--| model  |  optimizer | weight decay | learning rate | score |-->
<!--| :------------: | :--------------: | :--------------: | :--------------: | :--------------: |-->
<!--mixer_b16_224_miil_in21k | adam  | 1e-4 | 4e-4 | 90.5 |-->
<!--mixer_b16_224_miil_in21k | adam  | 1e-4 | 2e-4 | 91.1 |-->
<!--mixer_b16_224_miil_in21k | adamw | 1e-4 | 1e-4 | 90.9 |-->
<!--mixer_b16_224_miil_in21k | adamw | 1e-2 | 1e-4 | 90.9 |-->
<!--mixer_b16_224_miil_in21k | sgd   | 1e-4 | 1e-4 | 92.4 |-->
<!--|   |   |   |   |  |-->
<!--mixer_b16_224_in21k | adam  | 1e-4 | 4e-4 | 82.6 |-->
<!--mixer_b16_224_in21k | adam  | 1e-4 | 2e-4 | 84.0 |-->
<!--mixer_b16_224_in21k | adamw | 1e-4 | 2e-4 | 84.4 |-->
<!--mixer_b16_224_in21k | adamw | 1e-2 | 2e-4 | 84.7 |-->
<!--mixer_b16_224_in21k | sgd   | 1e-4 | 2e-4 | 91.7 |-->

|  Optimizer | Weight decay | Learning rate | Official pretrain Mixer-B-16 score |  Miil pretrain Mixer-B-16 score |
| :--------------: | :--------------: | :--------------: | :--------------: | :--------------: |
| adam  | 1e-4 | 4e-4 | 82.6 | **90.5** (+7.9) |
| adam  | 1e-4 | 2e-4 | 84.0 | **91.1** (+7.1) |
| adamw | 1e-4 | 2e-4 | 84.4 | **90.9** (+6.5) |
| adamw | 1e-2 | 2e-4 | 84.7 | **90.9** (+6.2) |
| sgd   | 1e-4 | 2e-4 | 91.7 | **92.4** (+0.7) |


We can see that miil pretraining reaches almost the same accuracy for all hyper-parameters, while the official pretraining suffers from a major drop in accuracy for adam and adamw.
For every hyper-parameter tested, miil pretraining achieves better accuracy.