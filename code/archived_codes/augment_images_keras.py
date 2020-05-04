import shutil
import os
import time
import tensorflow as tf
import numpy as np
from PIL import Image
import threading
from IPython.display import clear_output

# Note start time
start = time.time()

# Class labels
CLASSES = 'cdnp'

# Image size
IMG_SIZE = 256

# Training proportion
train_prop = 0.8

# Folder paths
FOLDER_PREFIX = f'/content/{IMG_SIZE}'
RAW_FOLDER = f'{FOLDER_PREFIX}/raw'
SPLIT_FOLDER = f'{FOLDER_PREFIX}/split'
NO_DISTORT_AUG_FOLDER = f'{FOLDER_PREFIX}/no_dist_aug'
AUG_FOLDER = f'{FOLDER_PREFIX}/aug'
TRAIN_FOLDER = f'{FOLDER_PREFIX}/train'
TEST_FOLDER = f'{FOLDER_PREFIX}/test'

# For all generated folders
for folder in [SPLIT_FOLDER, NO_DISTORT_AUG_FOLDER, AUG_FOLDER, 
               TRAIN_FOLDER, TEST_FOLDER]:

  # If the folder already exists
  if os.path.exists(folder):
    
    # Delete the folder
    shutil.rmtree(folder, onerror=lambda _, __, ___: _)

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
# to be augmented to get the input data close to 20,000 samples
aug_ratio = {class_: 20_000 // (num_samples[class_] * 8) for class_ in CLASSES}

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

# Function for thread that will generate the augmented images
def augment_imgs(class_):

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

        # Calculate files in the folder
        l = len(os.listdir(f'{AUG_FOLDER}/{class_}'))

        # Print progress
        thread_msg[class_] = f'Completed {l} / 20000'

        # If the augmentation ratio has been achieved
        if l == (aug_ratio[class_] * file_num):
          break

  # Split point between train and test
  train_test_split = 16_000

  # Print progress message
  thread_msg[class_] = 'Generating training images'

  # Scan the aug folder
  with os.scandir(f'{AUG_FOLDER}/{class_}') as folder:

    # Iterate for all files in the folder
    for i, file in enumerate(folder):

      # For training proportion, copy to train folder
      if i < train_test_split:
        shutil.copy(src=file.path, dst=f'{TRAIN_FOLDER}/{class_}')

      # For testing proportion, copy to test folder
      else:
        shutil.copy(src=file.path, dst=f'{TEST_FOLDER}/{class_}')

        # Print progress message
        thread_msg[class_] = 'Generating testing images'

  # Set the finish flag to true and progress message to finished
  thread_msg[class_] = 'Thread work finished'
  time.sleep(1)
  thread_finished[class_] = True

# Flag and progress message to indicate the status of threads
thread_finished = {class_: False for class_ in CLASSES}
thread_msg = {class_: 'Thread not started' for class_ in CLASSES}

# Start a thread for each class
for class_ in CLASSES:
  threading.Thread(target=augment_imgs, args=(class_, )).start()

# Main thread will wait for all threads to finish
while not all(thread_finished.values()):
  time.sleep(1)
  clear_output()
  print(*[f'{class_}: {thread_msg[class_]}' for class_ in CLASSES], sep='\n')

# This is the final count of images generated
gen_img_cnt = {class_: aug_ratio[class_] * num_samples[class_] * 8 for class_ in CLASSES}

# Print stats
print('Original images count')
print(num_samples)
print('Generated images count:')
print(gen_img_cnt)

# Archive the IMG_SIZE folder and copy it to drive
shutil.make_archive(f'{IMG_SIZE}', 'zip', f'{IMG_SIZE}')
shutil.copy(src=f'/content/{IMG_SIZE}.zip', dst='/content/drive/My Drive')

# Note end time
end = time.time()

# Calculate time difference
exec_time = int(end - start)

# Print execution time
print(f'Total execution time: {exec_time}s (' + \
      f'{exec_time // 3600}h {(exec_time // 60) % 60}m {exec_time % 60}s)')