from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import cv2

class CartoonProcessor:
    def __init__(self, generator_path="models/generator_epoch100_model.h5", discriminator_path="models/discriminator_epoch100_model.h5"):
        """
        Initializes class with generator and (optional) discriminator model paths.

        :param generator_path: Path of saved generator model (.h5).
        :param discriminator_path: Path of saved discriminator model (.h5, optional).
        """
        self.generator = load_model(generator_path)
        self.discriminator = load_model(discriminator_path) if discriminator_path else None
        print("succes:: Model loaded !")

    @staticmethod
    def preprocess_image(image_path, target_size=(256, 256)):
        """
        Preprocess an image for inference.

        :param image_path: Path of the image to be preprocessed.
        :param target_size: Target size for the image (default 256x256).
        :return: Image pre-processed as tensor.
        """
        image = Image.open(image_path).convert('RGB')
        image = image.resize(target_size)
        image = np.array(image).astype('float32') / 127.5 - 1  # Normalisation [-1, 1]
        return image[np.newaxis, ...]

    @staticmethod
    def preprocess_frame(frame, target_size=(256, 256)):
        """
        Pre-process a frame to make it compatible with the template.

        :param frame: Video frame (image).
        :param target_size: Target size for template (default 256x256).
        :return: Pre-processed frame.
        """
        frame = cv2.resize(frame, target_size)  #  Resize
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert to RGB
        frame = (frame / 127.5) - 1.0  # Normalize to [-1, 1]
        return frame

    @staticmethod
    def postprocess_image(image):
        """
        Post-process the generated image for display or saving.

        :param image: Image generated by template.
        :return: Post-processed image in PIL format.
        """
        image = ((image + 1) / 2 * 255).astype(np.uint8)  # Re-normalization [0, 255]
        return Image.fromarray(image)

    @staticmethod
    def postprocess_frame(frame):
        """
        Post-process a frame generated by the model for display or saving.

        :param frame: Frame generated by the model.
        :return: Post-processed frame in OpenCV (BGR) format.
        """
        frame = ((frame + 1) * 127.5).astype(np.uint8)  # Re-normalization [0, 255]
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # Convert to RGB
        return frame

    def generate_cartoon_image(self, image_path, output_path=None):
        """
        Generate a cartoon image from a real image.

        :param image_path: Path to input image.
        :param output_path: Path to save generated image (optional).
        :return: Image generated in PIL format.
        """
        # Image pre-processing
        input_image = self.preprocess_image(image_path)

        # Generate cartoon image
        generated_image = self.generator.predict(input_image, verbose=0)
        cartoon_image = self.postprocess_image(generated_image.squeeze())

        # Save the generated image if a path is provided
        if output_path:
            cartoon_image.save(output_path)
            print(f"Cartoon image saved as : {output_path}")

        return cartoon_image

    def generate_cartoon_video(self, input_video_path, output_video_path, target_size=(256, 256)):
        """
        Transform a video into a cartoon video.

        :param input_video_path: Input video path.
        :param output_video_path: Output video path.
        :param target_size: Target size for template (default 256x256).
        """
        # Load input video
        cap = cv2.VideoCapture(input_video_path)

        # Check that the input video is loaded correctly
        if not cap.isOpened():
            print("Error : Impossible to load the video.")
            return

        # Retrieve video properties
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Configure output video
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

        # Check that output video is correctly configured
        if not out.isOpened():
            print("Error : Impossible to create a ouptut video.")
            cap.release()
            return

        print(f"Transformation of {input_video_path} to {output_video_path}")
        print(f"Dimensions : {frame_width}x{frame_height}, FPS : {fps}, Total Frames : {total_frames}")

        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Resize the frame to the size expected by the model
            original_size = (frame.shape[1], frame.shape[0])  # Dimensions originales
            preprocessed_frame = self.preprocess_frame(frame, target_size)

            # Add a batch dimension for model input
            input_data = np.expand_dims(preprocessed_frame, axis=0)

            # Generate the cartoon frame
            generated_frame = self.generator.predict(input_data, verbose=0)
            cartoon_frame = self.postprocess_frame(generated_frame[0])

            # Resize the generated frame to its original size
            cartoon_frame = cv2.resize(cartoon_frame, original_size)

            # Add cartoon frame to output video
            out.write(cartoon_frame)

            frame_count += 1
            print(f"Frame {frame_count}/{total_frames} traitée.", end="\r")

        # Freeing up resources
        cap.release()
        out.release()
        print("\nTransformation completed !")
