# Animating Doodles with Autoencoders

This repo contains the code for my blog post on creating animations of doodles using autoencoders trained on synthetic data.

## Requirements

* pytorch 0.4
* [writefile-run](https://pypi.org/project/writefile-run/) `pip install writefile-run`
* [pytorch-utils](https://github.com/rajatvd/PytorchUtils) - clone this repo and follow the instructions

## The notebooks

* `modules.ipynb` and `autoenc_models.ipynb` contain the code for building the pytorch models.
* `lines_dataset.ipynb` contains the code for making the synthetic dataset.
* Run `train.ipynb` after setting a save directory name to train a CoordConv autoencoder with a latent space size of 64. You can change this and try out other models as well.
* Play around with a trained model to make gifs using the `analysis.ipynb` notebook.
