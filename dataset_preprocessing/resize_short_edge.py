import cv2
from multiprocessing.dummy import Pool as ThreadPool
from functools import partial
from shutil import copyfile
import os
import errno
import numpy as np

def resize_image_short_edge(inputFileName, output_size=256, input_str='imagenet21k_new', output_str='imagenet21k_resized_new'):
    try:
        out_path = inputFileName.replace(input_str, output_str)

        if not os.path.exists(os.path.dirname(out_path)):
            try:
                os.makedirs(os.path.dirname(out_path))
            except OSError as exc:  # Guard against race condition
                print("OSError ",inputFileName)
                if exc.errno != errno.EEXIST:
                    raise

        assert out_path != inputFileName
        im = cv2.imread(inputFileName)
        shape = im.shape
        max_dim = max(shape[0], shape[1])
        if max_dim < output_size:
            copyfile(inputFileName, out_path)
        else:
            im_shape = im.shape
            im_size_min = np.min(im_shape[0:2])
            im_size_max = np.max(im_shape[0:2])
            im_scale = float(output_size) / float(im_size_min)
            im_resize = cv2.resize(im, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_LINEAR)
            cv2.imwrite(out_path, im_resize)

    except:
        print("general failure ",inputFileName)


def main():
    path = '/mnt/imagenet21k_new/' # might need to edit this
    import os
    from glob import glob
    print("scanning files...")
    files = [y for x in os.walk(path) for y in glob(os.path.join(x[0], '*.*'))]
    print("done, start resizing")

    pool = ThreadPool(8)
    resize_image_fun = partial(resize_image_short_edge, input_str='imagenet21k_new', output_str='imagenet21k_resized_new')
    pool.map(resize_image_fun, files)


if __name__ == '__main__':
    main()