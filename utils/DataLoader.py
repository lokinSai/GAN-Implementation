import tensorflow_datasets as tfds


class DataLoader():
    def get_celeb_dataset(self, batch_size, buffer_size):
        # Construct a tf.data.Dataset
        # dataset = tfds.load('celeb_a', split='train', shuffle_files=True)
        # dataset = dataset.shuffle(buffer_size).batch(batch_size)
        return dataset
