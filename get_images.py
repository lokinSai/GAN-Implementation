from PIL import Image
from glob import glob
import os
import numpy as np
from tqdm import tqdm
import h5py
import tensorflow as tf

def process_image(fname, w=64, h=64):
    img = Image.open(fname)
    img_arr = np.array(img.resize((w,h)))
    # normalize between -1 ~ 1
    img_arr = (img_arr - 127.5) / 127.5
    return img_arr

# assuming you create a subset of images in /mini-data repository
def get_images(img_dir='mini_data'):
    filenames = glob(os.path.join(img_dir, '*.jpg'))
    w, h = 64, 64
    data = np.zeros((len(filenames), w, h, 3), dtype=np.float)
    for i, fname in tqdm(enumerate(filenames)):
        img_arr = process_image(fname)
        data[i] = img_arr
    return data

class generator:
    def __init__(self, filename):
        self.filename = filename

    def __call__(self):
        with h5py.File(self.filename, 'r') as f:
            for im in f['celeba']:
                yield im

def get_h5_images(filename='celeba_dataset.h5'):
    gen = generator(filename)
    dataset = tf.data.Dataset.from_generator(gen, tf.float32)
    return dataset

get_h5_images()
# get_images()
