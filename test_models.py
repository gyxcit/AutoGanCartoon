from cartoon_effects import CartoonProcessor

#load my model
processor = CartoonProcessor()

# Transform an image
input_image_path = "test_gen/72.jpg" #path ot your image to cartoonize
output_image_path = "cartoon_image.jpg" #name of your cartoon image + extension
cartoon_image = processor.generate_cartoon_image(input_image_path, output_image_path)
cartoon_image.show()


#transform video
input_video_path = "test_gen/test.mp4" #path ot your video to cartoonize
output_video_path = "cartoon_video2.mp4" #name of your cartoon video + extension
processor.generate_cartoon_video(input_video_path, output_video_path)