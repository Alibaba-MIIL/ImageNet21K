import os
import urllib
from argparse import Namespace
import torch
from PIL import Image
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from src_files.semantic.semantics import ImageNet21kSemanticSoftmax
import timm

############### Downloading metadata ##############
print("downloading metadata...")
url, filename = (
    "https://miil-public-eu.oss-eu-central-1.aliyuncs.com/model-zoo/ImageNet_21K_P/resources/imagenet21k_tree.pth",
    "imagenet21k_tree.pth")
if not os.path.isfile(filename):
    urllib.request.urlretrieve(url, filename)
args = Namespace()
args.tree_path = filename
semantic_softmax_processor = ImageNet21kSemanticSoftmax(args)
print("done")

############### Loading (ViT) model from timm package ##############
print("initilizing model...")
model = timm.create_model('vit_base_patch16_224_miil_in21k', pretrained=True)
model.eval()
config = resolve_data_config({}, model=model)
transform = create_transform(**config)
print("done")

############## Loading sample image ##############
print("downloading sample image...")
url, filename = ("https://github.com/pytorch/hub/raw/master/images/dog.jpg", "dog.jpg")
if not os.path.isfile(filename):
    urllib.request.urlretrieve(url, filename)
img = Image.open(filename).convert('RGB')
tensor = transform(img).unsqueeze(0)  # transform and add batch dimension
print("done")

############## Doing semantic inference ##############
print("doing semantic infernce...")
labels = []
with torch.no_grad():
    logits = model(tensor)
    semantic_logit_list = semantic_softmax_processor.split_logits_to_semantic_logits(logits)

    # scanning hirarchy_level_list
    for i in range(len(semantic_logit_list)):
        logits_i = semantic_logit_list[i]

        # generate probs
        probabilities = torch.nn.functional.softmax(logits_i[0], dim=0)
        top1_prob, top1_id = torch.topk(probabilities, 1)

        if top1_prob > 0.5:
            top_class_number = semantic_softmax_processor.hierarchy_indices_list[i][top1_id[0]]
            top_class_name = semantic_softmax_processor.tree['class_list'][top_class_number]
            top_class_description = semantic_softmax_processor.tree['class_description'][top_class_name]
            labels.append(top_class_description)

print("labels found {}.".format(labels))

############## Visualization ##############
import matplotlib
import os
import numpy as np

if os.name == 'nt':
    matplotlib.use('TkAgg')
else:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.imshow(img)
plt.axis('off')
plt.title('Semantic labels found: \n {}'.format(np.array(labels)))
plt.show()
