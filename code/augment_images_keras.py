import shutil
import os
import time
import tensorflow as tf
import numpy as np
from PIL import Image

# Note start time
start = time.time()

# Class labels
CLASSES = 'cdnp'

# Image size
IMG_SIZE = 256

# Folder paths
FOLDER_PREFIX = 'F:/github/Drones_For_Structural_Audit/dataset/internal'
RAW_FOLDER = f'{FOLDER_PREFIX}/{IMG_SIZE}_raw'
SPLIT_FOLDER = f'{FOLDER_PREFIX}/{IMG_SIZE}_split'
NO_DISTORT_AUG_FOLDER = f'{FOLDER_PREFIX}/{IMG_SIZE}_no_dist_aug'
AUG_FOLDER = f'{FOLDER_PREFIX}/{IMG_SIZE}_aug'

# For all 3 folders
for folder in [SPLIT_FOLDER, NO_DISTORT_AUG_FOLDER, AUG_FOLDER]:

  # If the folder already exists
  if os.path.exists(folder):
    
    # Delete the folder
    shutil.rmtree(folder, onerror=lambda _: _)

    # Wait for 1 second
    time.sleep(1)

  # Make empty folder again
  os.mkdir(folder)

  # Make class label sub folders
  for class_ in CLASSES:
    os.mkdir(f'{folder}/{class_}')

# Initialize count of samples in each class to 0
num_samples = {class_: 0 for class_ in CLASSES}

# Scan the raw samples folder
with os.scandir(RAW_FOLDER) as folder:

  # Iterate for all files in the folder
  for file in folder:

    # Extract the label number 0123 from the file name
    label = int(file.name.split('_')[3][0])

    # Increment the count of sample of this class by 1
    num_samples[CLASSES[label]] += 1

    # Copy the file to label folder inside the split folder
    shutil.copy(src=file.path, dst=f'{SPLIT_FOLDER}/{CLASSES[label]}')

    # Open this image file
    img = Image.open(file.path)

    # Save the original image in the no distortion aug folder
    img.save(f'{NO_DISTORT_AUG_FOLDER}/{CLASSES[label]}/{file.name[:-4]}_0deg.jpg')

    # For all non distorting augmentation operations and their name suffix
    for op, suffix in {
                        Image.ROTATE_90: '90deg', Image.ROTATE_180: '180deg',
                        Image.ROTATE_270: '270deg', Image.FLIP_TOP_BOTTOM: 'verti',
                        Image.FLIP_LEFT_RIGHT: 'hori', Image.TRANSPOSE: 'transpose',
                        Image.TRANSVERSE: 'transverse'
                       }.items():
      
      # Perform operation and save the file with proper suffix
      img.transpose(op)\
        .save(f'{NO_DISTORT_AUG_FOLDER}/{CLASSES[label]}/{file.name[:-4]}_{suffix}.jpg')

# The augmentation ratio is the factor by which the non distorted images need
# to be augmented to get the input data close to 10,000 samples
aug_ratio = {class_: 10_000 // (num_samples[class_] * 8) for class_ in CLASSES}

# Keras data generator which performs distorting
# augmentations as specified in the parameter
data_gen = tf.keras.preprocessing.image.ImageDataGenerator(
  zca_whitening = True,
  rotation_range = 44,
  width_shift_range = 0.2,
  height_shift_range = 0.2,
  brightness_range = [0.8, 1.2],
  shear_range = 0.2,
  zoom_range = 0.2,
)

# For all class labels
for class_ in CLASSES:

  # Scan the no distortion augmentation folder
  with os.scandir(f'{NO_DISTORT_AUG_FOLDER}/{class_}') as folder:

    # Iterate for all files in the folder
    for file_num, file in enumerate(folder, start=1):

      # Load the pixel array of the image file
      img = np.asarray(Image.open(file.path)).reshape((1, IMG_SIZE, IMG_SIZE, 3))

      # Start the augmentation process for the current image
      for _ in data_gen.flow(
                 x = img,
                 batch_size = 1,
                 shuffle = False,
                 save_to_dir = f'{AUG_FOLDER}/{class_}',
                 save_prefix = file.name[:-4],
                 save_format = 'jpeg'
               ):

        # If the augmentation ration has been achieved
        if len(os.listdir(f'{AUG_FOLDER}/{class_}')) == (aug_ratio[class_] * file_num):
          break

# Note end time
end = time.time()

# Calculate time difference
exec_time = int(end - start)

print(f'Total execution time: {exec_time}s (' + \
      f'{exec_time // 3600}h {(exec_time // 60) % 60}m {exec_time % 60}s)')