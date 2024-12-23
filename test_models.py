from cartoon_effects import CartoonProcessor

#load my model
processor = CartoonProcessor()

# Transform an image
"""
input_image_path = "test_gen/72.jpg"
output_image_path = "cartoon_image.jpg"
cartoon_image = processor.generate_cartoon_image(input_image_path, output_image_path)
cartoon_image.show()
"""

#transform tape

input_video_path = "test_gen/test.mp4"
output_video_path = "cartoon_video2.mp4"
processor.generate_cartoon_video(input_video_path, output_video_path)