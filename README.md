# AutoGanCartoon
Pour l'entraînement du modèle les fichiers à utiliser sont: split_dataset.ipynb et test_model_pix2pix.ipynb.

Les fichiers cartoon_modelPix2Pix.py et cartoon_model_pix2pix.ipynb sont à supprimer.

Les librairies utilisées sont les suivantes :
- import tensorflow as tf
- from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, BatchNormalization, LeakyReLU, Dropout, ReLU, Concatenate
- from tensorflow.keras.models import Model
- import os
- import numpy as np
- from tensorflow.keras.optimizers import Adam
- import glob
- from PIL import Image
- from skimage.metrics import structural_similarity as ssim
- from skimage.metrics import peak_signal_noise_ratio as psnr


