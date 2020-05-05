import os
import shutil
from PIL import Image
import random

# Folder paths
INPUT_FOLDER = 'F:/github/Drones_For_Structural_Audit/dataset/internal/grid/renamed_filtered_samples'
OUTPUT_FOLDER = 'F:/github/Drones_For_Structural_Audit/dataset/internal/grid/no_dist'

# Class labels
CLASSES = 'cdnp'

# Image size
IMG_SIZE = 256

# If the folder already exists, delete the folder
if os.path.exists(OUTPUT_FOLDER):
  shutil.rmtree(OUTPUT_FOLDER, onerror=lambda a, b, c: 0)
  
# Make the output folder and subfolders
os.mkdir(OUTPUT_FOLDER)
for class_ in CLASSES:
  os.mkdir(f'{OUTPUT_FOLDER}/{class_}')

# For all files in the input folder
for class_ in CLASSES:

  # Image name counter
  img_cntr = 0

  files = random.sample(os.listdir(f'{INPUT_FOLDER}/{class_}'), 625)

  for file in files:

    # Open the image file
    img = Image.open(f'{INPUT_FOLDER}/{class_}/{file}')

    # Scale the image
    scaled_img = img.resize((IMG_SIZE, IMG_SIZE))

    # Save the scaled image to output folder
    scaled_img.save(f'{OUTPUT_FOLDER}/{class_}/{img_cntr:04}.jpg')
    img_cntr += 1

    # For all non distorting augmentations
    for op in [Image.ROTATE_90, Image.ROTATE_180, Image.ROTATE_270,
                Image.FLIP_LEFT_RIGHT, Image.FLIP_TOP_BOTTOM,
                Image.TRANSPOSE, Image.TRANSVERSE]:

      # Apply the transformation
      aug_img = scaled_img.transpose(op)

      # Save the augmented image to the output folder
      aug_img.save(f'{OUTPUT_FOLDER}/{class_}/{img_cntr:04}.jpg')
      img_cntr += 1