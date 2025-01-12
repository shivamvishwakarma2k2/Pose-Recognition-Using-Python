# Importing the function to process images from the pose recognition module
from pose_recoginze_and_estimate import process_images

num_images = 27 # Defining the number of images to process

# Fetching a list of image from image Dir paths based on the number of images
image_paths = [f"images/image-{i}.png" for i in range(1, num_images + 1)]

# Invoking process_images function with image paths with pose estimation value (True, Flase)
process_images(image_paths, pose_est=True)