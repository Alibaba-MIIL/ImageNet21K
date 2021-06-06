## Obtaining ImageNet-21K-P

### Fall 11 Version
For a fair comparison to previous works, the article results are based on 'fall11_whole.tar' original release of ImageNet-21K.
The file contains all original images and classes, at full resolution. Its size is 1.31 TB.

For preprocessing ImageNet-21K-P from the original dataset (see the article for more details), you can use following [script](./processing_script.sh).
Make sure you update the target directory in the script, and also the file resize.py.

After you finish the preprocessing, the variant of ImageNet-21K-P based on fall1-release should contain:
"train set contains 11797632 samples, test set contains 561052 samples. 11221 classes"

### Winter 21 Version
Another way to obtain ImageNet-21K-P is via the [official ImageNet site](https://image-net.org/download.php), that now provides a direct link to the processed dataset, based on 'winter21_whole.tar.gz' file.
Compared to earlier releases of ImageNet-21K, the winter21 version removed a small number of classes and samples.

This variant of ImageNet-21K-P, based on winter21-release, is a dataset with:
"train set contains 11060223 samples, test set contains 522500 samples. 10450 classes".