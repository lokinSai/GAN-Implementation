import os
import time
import functools

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tensorflow import random_normal_initializer
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Input, Model, layers
from tensorflow.keras.layers import Dense, LeakyReLU, BatchNormalization, \
    Conv2D, Conv2DTranspose, Reshape, Flatten, Dropout


from utils import *

class InfoGAN():
    def __init__(self, buffer_size=6000, batch_size=32, epochs=8000,
                 noise_dim=100, n_control_cat=4, filename='celeba_dataset.h5', key='celeba'):
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.epochs = epochs
        self.noise_dim = noise_dim
        self.n_control_cat = n_control_cat
        self.train_dataset = ImageUtil(filename=filename, key=key).get_h5_images()
        self.train_dataset = self.train_dataset.shuffle(buffer_size).batch(batch_size)

    def _make_generator_model(self):
        # weight initialization
        init = random_normal_initializer(stddev=0.02)
        # image generator input
        in_noise = Input(shape=(self.noise_dim + self.n_control_cat,))
        gen = Dense(16*16*256, use_bias=False, kernel_initializer=init)(in_noise)
        gen = BatchNormalization()(gen)
        gen = LeakyReLU(alpha=0.1)(gen)
        gen = Dropout(0.3)(gen)

        gen = Reshape((16, 16, 256))(gen)

        gen = Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same',
            use_bias=False, kernel_initializer=init)(gen)
        gen = BatchNormalization()(gen)
        gen = LeakyReLU(alpha=0.1)(gen)

        gen = Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same',
            use_bias=False, kernel_initializer=init)(gen)
        gen = BatchNormalization()(gen)
        gen = LeakyReLU(alpha=0.1)(gen)

        out_layer = Conv2DTranspose(3, (5, 5), strides=(2, 2), padding='same',
            use_bias=False, kernel_initializer=init, activation='tanh')(gen)

        model = Model(inputs=in_noise, outputs=out_layer)

        return model

    def _make_discriminator_model(self):
        init = tf.random_normal_initializer(stddev=0.02)

        in_image = Input(shape=(64,64,3))
        d = Conv2D(64, (5,5), strides=(2,2), padding='same', kernel_initializer=init)(in_image)
        d = LeakyReLU(alpha=0.1)(d)
        d = Dropout(0.3)(d)

        d = Conv2D(128, (5,5), strides=(2,2), padding='same', kernel_initializer=init)(d)
        d = LeakyReLU(alpha=0.1)(d)
        d = Dropout(0.3)(d)

        d = Flatten()(d)

        out_classifier = Dense(1, activation='sigmoid')(d)

        d_model = Model(inputs=in_image, outputs=out_classifier)
        d_model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0002, beta_1=0.5))

        q = Dense(128)(d)
        q = BatchNormalization()(q)
        q = LeakyReLU(alpha=0.1)(q)
        out_codes = Dense(self.n_control_cat, activation='softmax')(q)

        q_model = Model(inputs=in_image, outputs=out_codes)

        return d_model, q_model

    def _make_gan_model(self, g_model, d_model, q_model):
        # make weights in the discriminator (some shared with the q model) as not trainable
        d_model.trainable = False
        # connect g outputs to d inputs
        d_output = d_model(g_model.output)
        # connect g outputs to q inputs
        q_output = q_model(g_model.output)
        # define composite model
        model = Model(inputs=g_model.input, outputs=[d_output, q_output])
        # compile model
        opt = Adam(lr=0.0002, beta_1=0.5)
        model.compile(loss=['binary_crossentropy', 'categorical_crossentropy'], optimizer=opt)
        return model

    def _generate_noise_and_control(self):
        noise = tf.random.normal([self.batch_size, self.noise_dim])
        rand_cat = np.random.randint(0, self.n_control_cat, size=self.batch_size)
        control_cat = tf.keras.utils.to_categorical(rand_cat, num_classes=self.n_control_cat)
        noise = tf.concat([noise, control_cat], axis=1)
        return noise, control_cat

    def _infoGan_train_step(self, images, g_model, d_model, gan_model):
        noise, control_cat = self._generate_noise_and_control()

        d_loss_true = d_model.train_on_batch(images, np.ones((self.batch_size, 1)))

        generated_images = g_model.predict(noise)
        d_loss_fake = d_model.train_on_batch(generated_images, np.zeros((self.batch_size, 1)))

        # inverted labels for fake images
        # train g_model & q_model at the same time
        _, g_1, g_2 = gan_model.train_on_batch(noise, [np.ones((self.batch_size, 1)), control_cat])
        return d_loss_true, d_loss_fake, g_1, g_2

    def train(self):
        g_model = self._make_generator_model()
        d_model, q_model = self._make_discriminator_model()
        gan_model = self._make_gan_model(g_model, d_model, q_model)

        image_helper = ImageHelper(noise_dim=self.noise_dim)

        for epoch in range(self.epochs):
            start = time.time()
            for image_batch in self.train_dataset:
                if image_batch.shape[0] < self.batch_size:
                    break
                self._infoGan_train_step(image_batch, g_model, d_model, gan_model)

            if (epoch + 1) % 30 == 0:
                # checkpoint.save(file_prefix=checkpoint_prefix)
                image_helper.generate_and_save_images_control_cat(g_model, epoch+1, self.n_control_cat)

            print('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))

        # Generate after the final epoch
        image_helper.generate_and_save_images_control_cat(g_model, self.epochs, self.n_control_cat)
