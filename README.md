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

# Projet de cartonisation des images 
Pour le projet la première étape a été de créer un algorithme qui nous permet de cartoonizer les images en utilisant des techniques de computer vision. Cet algorithme se trouve dans le fichier cartoonization.py, grâce à lui nous avons pu cartooniser nos images et créer la base de connaissance pour l'entraînement du modèle. Pour la deuxième étape qui consistait à créer un modèle de Pix2Pix qui est modèle de GAN (réseaux antagonistes génératifs), pour se faire nous avon stout d'abord separer nos données de notre base de connaissances en données d'entraînement et de test du modèle. Cette étape est situé dans le fichier split_dataset. Ensuite nous avons entraîné notre modèle, dans le fichier test_model_pix2pix.ipynb on retrouve la création du modèle et et son entraînement.
