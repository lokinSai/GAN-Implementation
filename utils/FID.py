import numpy as np
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.datasets.mnist import load_data
from skimage.transform import resize
from scipy.linalg import sqrtm


class FID():
    def __init__(self, input_shape=(299, 299, 3)):
        self.input_shape = input_shape
        self.model = InceptionV3(include_top=False, pooling='avg', input_shape=self.input_shape)

    # scale an array of images to a new size
    def _scale_images(self, images, new_shape):
        images_list = []

        for image in images:
            new_image = resize(image, new_shape, 0)
            images_list.append(new_image)

        return np.asarray(images_list)

    # calculate FID
    def calculate_fid(self, images1, images2):
        print('Loaded', images1.shape, images2.shape)  # For debug only
        images1 = images1.astype('float32')
        images2 = images2.astype('float32')

        images1 = self._scale_images(images1, self.input_shape)
        images2 = self._scale_images(images2, self.input_shape)
        print('Scaled', images1.shape, images2.shape)

        images1 = preprocess_input(images1)
        images2 = preprocess_input(images2)

        activations1 = self.model.predict(images1)
        activations2 = self.model.predict(images2)
        mean1, covariance1 = activations1.mean(axis=0), np.cov(activations1, rowvar=False)
        mean2, covariance2 = activations2.mean(axis=0), np.cov(activations2, rowvar=False)
        sum_squared_diff = np.sum((mean1 - mean2)**2.0)
        covmean = sqrtm(covariance1.dot(covariance2))

        if np.iscomplexobj(covmean):
            covmean = covmean.real

        fid = sum_squared_diff + np.trace(covariance1 + covariance2 - 2.0 * covmean)

        return fid
