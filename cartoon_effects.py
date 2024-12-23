from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import cv2

class CartoonProcessor:
    def __init__(self, generator_path="models/generator_epoch100_model.h5", discriminator_path="models/discriminator_epoch100_model.h5"):
        """
        Initialise la classe avec les chemins des modèles générateur et (optionnel) discriminateur.

        :param generator_path: Chemin du modèle générateur sauvegardé (.h5).
        :param discriminator_path: Chemin du modèle discriminateur sauvegardé (.h5, facultatif).
        """
        self.generator = load_model(generator_path)
        self.discriminator = load_model(discriminator_path) if discriminator_path else None
        print("Modèles chargés avec succès !")

    @staticmethod
    def preprocess_image(image_path, target_size=(256, 256)):
        """
        Prétraiter une image pour l'inférence.

        :param image_path: Chemin de l'image à prétraiter.
        :param target_size: Taille cible pour l'image (par défaut 256x256).
        :return: Image prétraitée sous forme de tenseur.
        """
        image = Image.open(image_path).convert('RGB')
        image = image.resize(target_size)
        image = np.array(image).astype('float32') / 127.5 - 1  # Normalisation [-1, 1]
        return image[np.newaxis, ...]

    @staticmethod
    def preprocess_frame(frame, target_size=(256, 256)):
        """
        Prétraiter une frame pour la rendre compatible avec le modèle.

        :param frame: Frame de la vidéo (image).
        :param target_size: Taille cible pour le modèle (par défaut 256x256).
        :return: Frame prétraitée.
        """
        frame = cv2.resize(frame, target_size)  # Redimensionner
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convertir en RGB
        frame = (frame / 127.5) - 1.0  # Normaliser en [-1, 1]
        return frame

    @staticmethod
    def postprocess_image(image):
        """
        Post-traiter l'image générée pour affichage ou sauvegarde.

        :param image: Image générée par le modèle.
        :return: Image post-traitée en format PIL.
        """
        image = ((image + 1) / 2 * 255).astype(np.uint8)  # Re-normalisation [0, 255]
        return Image.fromarray(image)

    @staticmethod
    def postprocess_frame(frame):
        """
        Post-traiter une frame générée par le modèle pour affichage ou sauvegarde.

        :param frame: Frame générée par le modèle.
        :return: Frame post-traitée en format OpenCV (BGR).
        """
        frame = ((frame + 1) * 127.5).astype(np.uint8)  # Re-normaliser en [0, 255]
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # Convertir en BGR
        return frame

    def generate_cartoon_image(self, image_path, output_path=None):
        """
        Générer une image cartoon à partir d'une image réelle.

        :param image_path: Chemin de l'image d'entrée.
        :param output_path: Chemin pour sauvegarder l'image générée (facultatif).
        :return: Image générée en format PIL.
        """
        # Prétraiter l'image
        input_image = self.preprocess_image(image_path)

        # Générer l'image cartoon
        generated_image = self.generator.predict(input_image, verbose=0)
        cartoon_image = self.postprocess_image(generated_image.squeeze())

        # Sauvegarder l'image générée si un chemin est fourni
        if output_path:
            cartoon_image.save(output_path)
            print(f"Image cartoon sauvegardée sous : {output_path}")

        return cartoon_image

    def generate_cartoon_video(self, input_video_path, output_video_path, target_size=(256, 256)):
        """
        Transformer une vidéo en vidéo cartoon.

        :param input_video_path: Chemin de la vidéo d'entrée.
        :param output_video_path: Chemin de la vidéo de sortie.
        :param target_size: Taille cible pour le modèle (par défaut 256x256).
        """
        # Charger la vidéo d'entrée
        cap = cv2.VideoCapture(input_video_path)

        # Vérifier si la vidéo d'entrée est bien chargée
        if not cap.isOpened():
            print("Erreur : Impossible de charger la vidéo d'entrée.")
            return

        # Récupérer les propriétés de la vidéo
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Configurer la vidéo de sortie
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec pour MP4
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

        # Vérifier si la vidéo de sortie est bien configurée
        if not out.isOpened():
            print("Erreur : Impossible de créer le fichier vidéo de sortie.")
            cap.release()
            return

        print(f"Transformation de {input_video_path} en {output_video_path}")
        print(f"Dimensions : {frame_width}x{frame_height}, FPS : {fps}, Total Frames : {total_frames}")

        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Redimensionner la frame à la taille attendue par le modèle
            original_size = (frame.shape[1], frame.shape[0])  # Dimensions originales
            preprocessed_frame = self.preprocess_frame(frame, target_size)

            # Ajouter une dimension batch pour l'entrée du modèle
            input_data = np.expand_dims(preprocessed_frame, axis=0)

            # Générer la frame cartoon
            generated_frame = self.generator.predict(input_data, verbose=0)
            cartoon_frame = self.postprocess_frame(generated_frame[0])

            # Redimensionner la frame générée à la taille originale
            cartoon_frame = cv2.resize(cartoon_frame, original_size)

            # Ajouter la frame cartoon à la vidéo de sortie
            out.write(cartoon_frame)

            frame_count += 1
            print(f"Frame {frame_count}/{total_frames} traitée.", end="\r")

        # Libérer les ressources
        cap.release()
        out.release()
        print("\nTransformation terminée !")
