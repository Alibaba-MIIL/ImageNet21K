import pickle
import zlib

files_train_set = './resources/datasets_imagenet21K_P_train_folder_index.pkl'
files_val_set = './resources/datasets_imagenet21K_P_val_folder_index.pkl'

# train files
with open(files_train_set, 'rb') as fp:
    data = fp.read()
keys_relative, classes, class_to_idx = pickle.loads(zlib.decompress(data))

file_list = []
for key in keys_relative:
    if '.JPEG' in key or '.PNG' in key:
        file_list.append(key)

print("number of files in train set: {}".format(len(file_list)))

# validation files
with open(files_val_set, 'rb') as fp:
    data = fp.read()
keys_relative, classes, class_to_idx = pickle.loads(zlib.decompress(data))

file_list = []
for key in keys_relative:
    if '.JPEG' in key or '.PNG' in key:
        file_list.append(key)

print("number of files in validation set: {}".format(len(file_list)))