import os
import time
import functools

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras import layers

from utils import *


class GAN():
    def __init__(self, buffer_size=6000, batch_size=1000, epochs=8000, noise_dim=100, filename='celeba_dataset.h5'):
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.epochs = epochs
        self.noise_dim = noise_dim
        self.train_dataset = ImageUtil(filename=filename, key='celeba').get_h5_images()
        self.train_dataset = self.train_dataset.shuffle(buffer_size).batch(batch_size)

    def _make_generator_model(self):
        model = tf.keras.Sequential()
        model.add(layers.Dense(16*16*256, use_bias=False, input_shape=(self.noise_dim,)))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())

        model.add(layers.Reshape((16, 16, 256)))
        assert model.output_shape == (None, 16, 16, 256)  # Note: None is the batch size

        model.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
        assert model.output_shape == (None, 16, 16, 128)
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())

        model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
        assert model.output_shape == (None, 32, 32, 64)
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())

        model.add(layers.Conv2DTranspose(3, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
        assert model.output_shape == (None, 64, 64, 3)

        return model

    def _make_discriminator_model(self):
        model = tf.keras.Sequential()
        model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=[64, 64, 3]))
        assert model.output_shape == (None, 32, 32, 64)
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(0.3))

        model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
        assert model.output_shape == (None, 16, 16, 128)
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(0.3))

        model.add(layers.Flatten())
        model.add(layers.Dense(1))

        return model

    def _discriminator_loss(self, cross_entropy, real_output, fake_output):
        real_loss = tf.reduce_mean(real_output)
        fake_loss = tf.reduce_mean(fake_output)
        total_loss = - real_loss + fake_loss
        return total_loss

    def _generator_loss(self, cross_entropy, fake_output):
        fake_loss_gan = - tf.reduce_mean(fake_output)
        return fake_loss_gan

    @tf.function
    def _train_step(self,
                    images,
                    generator,
                    discriminator,
                    generator_optimizer,
                    discriminator_optimizer,
                    cross_entropy,
                    batch_size):
        noise = tf.random.normal([batch_size, self.noise_dim])

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_images = generator(noise, training=True)

            real_output = discriminator(images, training=True)
            fake_output = discriminator(generated_images, training=True)
            gen_loss = self._generator_loss(cross_entropy, fake_output)
            disc_loss = self._discriminator_loss(cross_entropy, real_output, fake_output)

        gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

        generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
        discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

    def train(self, checkpoint_dir='./training_checkpoints'):
        generator = self._make_generator_model()
        generator_optimizer = tf.keras.optimizers.Adam(1e-4)
        discriminator = self._make_discriminator_model()
        discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

        cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                         discriminator_optimizer=discriminator_optimizer,
                                         generator=generator,
                                         discriminator=discriminator)

        checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
        image_helper = ImageHelper(noise_dim=self.noise_dim)

        for epoch in range(self.epochs):
            start = time.time()
            for image_batch in self.train_dataset:
                self._train_step(image_batch,
                                 generator,
                                 discriminator,
                                 generator_optimizer,
                                 discriminator_optimizer,
                                 cross_entropy,
                                 self.batch_size)

            if (epoch + 1) % 1000 == 0:
                checkpoint.save(file_prefix=checkpoint_prefix)
                image_helper.generate_and_save_images(generator, epoch+1)

            print('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))

        # Generate after the final epoch
        image_helper.generate_and_save_images(generator, self.epochs)
