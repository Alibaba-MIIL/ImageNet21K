## Obtaining ImageNet-21K-P

Note: usage of ImageNet-21K-P is subjected to *image-net.org* [terms of access](https://image-net.org/download.php)

### Fall 11 Version
For a fair comparison to previous works, the article results are based on 'fall11_whole.tar' original release of ImageNet-21K.
The file contains all the original images and classes of ImageNet-21K, at full resolution. Its size is 1.31 TB.

For preprocessing ImageNet-21K-P from the original dataset (see the article for more details), you can use following end-to-end [script](./processing_script.sh).
After you finish the preprocessing, the variant of ImageNet-21K-P, based on fall1 release, should include:
<br>
"train set contains 11797632 samples, test set contains 561052 samples. 11221 classes"

### Winter 21 Version
We collaborated with *image-net.org* to enable direct downloading of ImageNet-21K-P via the [official ImageNet site](https://image-net.org/download.php).  
<br>  
This variant of the processed dataset is based on 'winter21_whole.tar.gz' release of ImageNet-21K.
Compared to earlier releases of ImageNet-21K, the winter21 version removed a small number of classes and samples.

The variant of ImageNet-21K-P is a dataset with:
<br>
"train set contains 11060223 samples, test set contains 522500 samples. 10450 classes"
<br>
To train with winter21 version, use the relevant hierarchial tree:
```
--tree_path=./resources/winter21_imagenet21k_miil_tree.pth
```