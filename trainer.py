from GAN_model import GAN


def run():
    gan = GAN(epochs=10)
    gan.train()


if __name__ == '__main__':
    run()
