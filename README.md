# GAN Model
## Steps:
1. Install pip dependencies `> pip3 install -r requirements.txt`
2. Download [CelebA-Align-Dataset](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)
3. Unzip downloaded file and make sure you created a repository name "img_align_celeba" with training images inside
4. `> python3 dataset_to_h5.py`: to create h5 dataset file # Set mini_test flag to False for creating the whole dataset
5. `> python3 GAN_model.py`: to train basic GAN Model.

## Process:
For every 5 epoch, model will randomly generate few images under /images folder 
