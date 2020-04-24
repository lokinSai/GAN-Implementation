from GAN_model import GAN
from InfoGAN_model import InfoGAN

def run():
    gan = InfoGAN(batch_size=32, epochs=10, filename='cartoon_dataset.h5', key='cartoon')
    gan.train()

if __name__ == '__main__':
    run()
