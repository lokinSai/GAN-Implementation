# GAN Model

## Steps:

1. Install pip dependencies `> pip3 install -r requirements.txt`
2. Download [CelebA-Align-Dataset](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)
3. Unzip downloaded file and make sure you created a repository name "img_align_celeba" with training images inside
4. `> python3 dataset_to_h5.py`: to create h5 dataset file # Set mini_test flag to False for creating the whole dataset
5. `> python3 trainer.py`: to train basic GAN Model.
6. `> python3 fid_samples.py` to Calculate FID on the saved images.

## Steps to run DCGAN
1. You should have a directory named `img_align_celeba` containing the celebrity images, and the `fid_dump`, `images` folders should be empty. You should also have the `celeba_dataset.h5` file ready.
2. Run `python3 GAN_model.py`. This will generate samples for calculating FID in `fid_dump` folder, and save generated images at regular epochs in the `images` folder. You will also get the `epochs_vs_loss.jpg` graph.
3. Run `python3 fid_samples.py`. This will calculate Frechet Inception Distance from the samples saved in previous step and by picking random images from `img_align_celeba` folder. In the end this will generate a `FID_vs_epochs.jpg` file where it calculates the FID multiple times.

The `DCGAN.ipynb` downloads the h5 file from Google Drive and trains the generator. It does not calculate FID dur to the large size of dataset involved, but is just intended to show how the training works. Moreover it takes a long time to run on Collab and the results start making sense only after around 3000 epochs. Steps 1-3 should be followed to get all the results.

## Steps to run InfoGAN
Simply visit [InfoGAN.ipynb](https://github.com/lokinSai/GAN-Implementation/blob/master/InfoGAN.ipynb) and open with Google Colab, it should be able to train on cartoon dataset.


## Process:

For every 5 epoch, model will randomly generate few images under /images folder

## Notes:

### Frechet Inception Distance (FID):

To test the FID package on the CIFAR10 dataset:  
`> python3 test_fid.py`

The default input shape is (299, 299, 3) but can be easily changed by passing in a tuple to the FID class.
