import numpy as np
from tensorflow.keras.datasets import cifar10

import tensorflow as tf
# Fix to run on Aditya's server :
for gpu in tf.config.experimental.list_physical_devices('GPU') : tf.config.experimental.set_memory_growth(gpu, True)

from utils import FID

def main():
    (images1, _), (images2, _) = cifar10.load_data()
    np.random.shuffle(images1)

    images1 = images1[:100]
    images2 = images2[:100]

    print('Loaded', images1.shape, images2.shape)

    fid = FID().calculate_fid(images1, images2)
    print('FID: {:.3f}'.format(fid))


if __name__ == "__main__":
    main()
