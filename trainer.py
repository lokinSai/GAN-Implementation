from InfoGAN_model import InfoGAN

def run():
    gan = InfoGAN(batch_size=16, epochs=1000, n_control_cat=8, filename='cartoon_dataset.h5', key='cartoon')
    gan.train()

if __name__ == '__main__':
    run()
