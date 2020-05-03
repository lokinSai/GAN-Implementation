#!python3
import matplotlib; matplotlib.use('Agg')
import os, random
import numpy as np
from glob import glob
from PIL import Image
from tqdm import tqdm
from utils import *
import tensorflow as tf
import matplotlib.pyplot as plt
# Fix to run on Aditya's server :
for gpu in tf.config.experimental.list_physical_devices('GPU') : tf.config.experimental.set_memory_growth(gpu, True)


# Randomly sample n images and return array

def get(img_dir="./img_align_celeba", n=100):
    filenames = glob(os.path.join(img_dir, '*.jpg'))
    random.shuffle(filenames)
    new_files = filenames[0:n]
    w, h = 64, 64
    data = np.zeros((len(new_files), w, h, 3), dtype=np.float)
    for i, fname in tqdm(enumerate(new_files)):
        img_arr = process_image(fname)
        if img_arr.shape[2] == 3:
            data[i] = img_arr
        else :
            data[i] = img_arr[:,:,:3]
    return data

def process_image(fname, w=64, h=64):
    img = Image.open(fname)
    img_arr = np.array(img.resize((w,h)))
    # normalize between -1 ~ 1
    img_arr = (img_arr - 127.5) / 127.5
    return img_arr

if __name__ == '__main__':
    fig = plt.figure()
    for j in range(10):
        fid_array = []
        x = []
        samples = glob(os.path.join("fid_dump", '*.npy')) ; samples.sort()
        inception = FID()
        for name in samples:
            x_label = name.split("/")[1][:-4]
            x.append(int(x_label))
            fid_gen_imgs = np.load(name)
            fid_data_imgs = get( n=1000)
            fid = inception.calculate_fid(fid_gen_imgs, fid_data_imgs)
            print("FID for epoch ", x_label, " was ", fid)
            fid_array.append(fid)

        indices = np.argsort(x)
        # Plot
        plt.plot(np.array(x)[indices], np.array(fid_array)[indices] ,label='FID')
        plt.xlabel("epochs")
        plt.ylabel("FID")
        plt.title("Frechet Inception distance") ;

    plt.grid()
    plt.savefig('FID_graph.png')


