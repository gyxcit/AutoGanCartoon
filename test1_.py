from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

# Charger les modèles sauvegardés
generator = load_model("generator_model.h5")
discriminator = load_model("discriminator_model.h5")

def preprocess_image(image_path, target_size=(256, 256)):
    """
    Prétraiter une image pour l'inférence.
    """
    image = Image.open(image_path).convert('RGB')
    image = image.resize(target_size)
    image = np.array(image).astype('float32') / 127.5 - 1  # Normalisation [-1, 1]
    return image[np.newaxis, ...]

def postprocess_image(image):
    """
    Post-traiter l'image générée pour affichage ou sauvegarde.
    """
    image = ((image + 1) / 2 * 255).astype(np.uint8)  # Re-normalisation [0, 255]
    return Image.fromarray(image)


if __name__ == "__main__":
    input_image_path = "test_gen/65.jpg"
    input_image = preprocess_image(input_image_path)

    # Génération avec le modèle chargé
    generated_image = generator(input_image, training=False).numpy()
    output_image = postprocess_image(generated_image[0])  # Supprimer la dimension batch

    # Afficher ou sauvegarder l'image générée
    output_image.show()  # Afficher l'image
    output_image.save("cartoonified_image.jpg")