# AutoGanCartoon
## prerequis
### Installation
1. clone the repo
```bash
git clone https://github.com/gyxcit/AutoGanCartoon.git
```
2. create and activate a venv (on windows)
```bash
#create the env
python -m venv venv

#activate
.\venv\Script\activate
```

3. install requirements
```bash
pip install -r requirement.txt
```

### Test 
if you want to test directly, you can use the pretrain version we setup in the file **test.ipynb** or **test1_.py**

---
## Informations
# Projet de cartonisation des images 
Pour le projet la première étape a été de créer un algorithme qui nous permet de cartoonizer les images en utilisant des techniques de computer vision. Cet algorithme se trouve dans le fichier cartoonization.py, grâce à lui nous avons pu cartooniser nos images et créer la base de connaissance pour l'entraînement du modèle.
Pour la deuxième étape qui consistait à créer un modèle de Pix2Pix qui est modèle de GAN (réseaux antagonistes génératifs), pour se faire nous avons tout d'abord separer nos données de notre base de connaissances en données d'entraînement et de test du modèle. Cette étape est situé dans le fichier split_dataset.ipynb. Ensuite nous avons entraîné notre modèle, dans le fichier trainingGan.ipynb on y retrouve la création du modèle et et son entraînement.L'image de la learning curves nous permet de visualiser la courbe d'apprentissage de notre modèle et donc de la valider.
Enfin dans les fichiers test.ipynb et test1_.py nous retrouvons les différents test de notre modèle.
