import tensorflow as tf
import time
import functools
import matplotlib.pyplot as plt
import numpy as np
import os
from tensorflow.keras import layers
from utils.checkpoint import get_checkpoint
from utils.Generate_image import Image_helper
from get_images import get_h5_images

def make_generator_model(noise_dim):
    model = tf.keras.Sequential()
    model.add(layers.Dense(16*16*256, use_bias=False, input_shape=(noise_dim,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Reshape((16, 16, 256)))
    assert model.output_shape == (None, 16, 16, 256) # Note: None is the batch size

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

def make_discriminator_model():
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same',
                                     input_shape=[64, 64, 3]))
    assert model.output_shape == (None, 32, 32, 64)
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    assert model.output_shape == (None, 16, 16, 128)
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(1,activation="sigmoid"))

    return model

def discriminator_loss(cross_entropy, real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

def generator_loss(cross_entropy, fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

@tf.function
def train_step(images, generator, discriminator, \
               generator_optimizer, discriminator_optimizer, \
               cross_entropy, batch_size):
    noise = tf.random.normal([batch_size, noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)

        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)

        gen_loss = generator_loss(cross_entropy, fake_output)
        disc_loss = discriminator_loss(cross_entropy, real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

def train(dataset, epochs, batch_size, noise_dim, checkpoint_dir='./training_checkpoints'):
    generator = make_generator_model(noise_dim)
    generator_optimizer = tf.keras.optimizers.Adam(1e-4)
    discriminator = make_discriminator_model()
    discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)
    
    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    checkpoint = get_checkpoint(generator, discriminator, generator_optimizer, discriminator_optimizer, checkpoint_dir=checkpoint_dir)
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    image_helper = Image_helper(noise_dim)

    for epoch in range(epochs):
        start = time.time()
        for image_batch in dataset:
            train_step(image_batch, generator, discriminator, 
                       generator_optimizer, discriminator_optimizer, cross_entropy, batch_size)
        if (epoch + 1) % 5 == 0:
            # checkpoint.save(file_prefix = checkpoint_prefix)
            image_helper.generate_and_save_images(generator, epoch+1)

        print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))

    # Generate after the final epoch
    image_helper.generate_and_save_images(generator, epochs)

# def image_to_dataset(images, batch_size, buffer_size):
#     dataset = tf.data.Dataset.from_tensor_slices(images) \
#         .shuffle(buffer_size).batch(batch_size)
#     return dataset

if __name__ == '__main__':
    buffer_size = 6000
    batch_size = 1
    epochs = 400
    noise_dim = 100
    # images = get_images()
    # train_dataset = image_to_dataset(images, batch_size, buffer_size)
    train_dataset = get_h5_images()
    train_dataset = train_dataset.shuffle(buffer_size).batch(batch_size)
    # train_dataset = get_celeb_dataset(batch_size, buffer_size)
    train(train_dataset, epochs, batch_size, noise_dim)
