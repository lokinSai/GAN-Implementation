from PIL import Image
import h5py
import numpy as np
from glob import glob
import os
from tqdm import tqdm


def process_image(fname, w=64, h=64):
    img = Image.open(fname)
    img_arr = np.array(img.resize((w, h)))
    # normalize between -1 ~ 1
    img_arr = (img_arr - 127.5) / 127.5
    return img_arr


def main(mini_test=False):
    img_dir = 'img_align_celeba'
    filenames = glob(os.path.join(img_dir, '*.jpg'))
    size = len(filenames)
    if mini_test:
        size = 100

    w, h, dim = 64, 64, 3

    with h5py.File('celeba_dataset.h5', 'w') as f:
        dataset = f.create_dataset(name='celeba', shape=(size, w, h, dim), dtype=np.float32)
        for i, fname in tqdm(enumerate(filenames[:size])):
            dataset[i] = process_image(fname, w, h)


if __name__ == '__main__':
    main(mini_test=True)
