import os
import tensorflow as tf

def get_checkpoint(generator, discriminator, generator_optimizer, \
                   discriminator_optimizer, checkpoint_dir):
    checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                    discriminator_optimizer=discriminator_optimizer,
                                    generator=generator,
                                    discriminator=discriminator)
    return checkpoint
