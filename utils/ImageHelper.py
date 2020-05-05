import matplotlib.pyplot as plt
import tensorflow as tf
import os

class ImageHelper():
    def __init__(self, noise_dim, num_examples_to_generate=16, img_dir='images'):
        try:
            os.mkdir(img_dir)
        except OSError as error:
            pass
        self.img_dir = img_dir
        self.num_examples_to_generate = num_examples_to_generate
        self.noise_dim = noise_dim
        self.seed = tf.random.normal([num_examples_to_generate, noise_dim])

    def _generate_noise_and_control(self, n_control_cat, n_sample_per_category):
        noise = tf.random.normal([n_sample_per_category, self.noise_dim])
        # repeat same noise for each control category
        noise = tf.concat([noise for _ in range(n_control_cat)], axis=0)
        # generate images for each control category
        rand_cat = []
        for cat in range(n_control_cat):
            rand_cat.extend([cat for _ in range(n_sample_per_category)])
        control_cat = tf.keras.utils.to_categorical(rand_cat, num_classes=n_control_cat)
        noise = tf.concat([noise, control_cat], axis=1)
        return noise

    def generate_and_save_images(self, model, epoch):
        # Notice `training` is set to False.
        # This is so all layers run in inference mode (batchnorm).
        predictions = model(self.seed, training=False)

        fig = plt.figure(figsize=(4, 4))

        for i in range(predictions.shape[0]):
            plt.subplot(4, 4, i+1)
            plt.imshow((predictions[i, :, :, :] + 1) / 2.0)
            plt.axis('off')

        plt.savefig('{}/image_at_epoch_{:04d}.png'.format(self.img_dir, epoch))

    def generate_and_save_images_control_cat(self, model, epoch, n_control_cat, n_sample_per_category=4):
        predictions = model.predict(
            self._generate_noise_and_control(n_control_cat, n_sample_per_category))
        fig = plt.figure(figsize=(n_control_cat, n_sample_per_category))

        for i in range(predictions.shape[0]):
            plt.subplot(n_control_cat, n_sample_per_category, i+1)
            plt.imshow((predictions[i, :, :, :] + 1) / 2.0)
            plt.axis('off')

        plt.savefig('{}/image_at_epoch_{:04d}.png'.format(self.img_dir, epoch))
