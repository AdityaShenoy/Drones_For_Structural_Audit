import os
import shutil
import tensorflow as tf
import numpy as np
from PIL import Image
import random
import time
import threading
from IPython.display import clear_output

# Note starting time
start = time.time()

# Constants
IMG_SIZE = 256
KERAS_DISTORTION_SCALE = 2
TRAIN_SPLIT = 0.6
VALIDATE_SPLIT = 0.8
CLASSES = 'cdnp'
NUM_SAMPLES = 625

# Folder and file paths
DESKTOP = 'C:/Users/admin/Desktop/content'
CONTENT = '/content' if not os.path.exists(DESKTOP) else DESKTOP
DRIVE = f'{CONTENT}/drive/My Drive'
DRIVE_RENAMED_FILTERED_SAMPLES_ZIP = f'{DRIVE}/renamed_filtered_samples.zip'
DATASET = f'{CONTENT}/dataset'
RENAMED_FILTERED_SAMPLES = f'{DATASET}/renamed_filtered_samples'
RENAMED_FILTERED_SAMPLES_ZIP = f'{RENAMED_FILTERED_SAMPLES}.zip'
RAW = f'{DATASET}/raw'
NO_DIST = f'{DATASET}/no_dist'
TRAIN = f'{DATASET}/train'
VALIDATE = f'{DATASET}/validate'
TEST = f'{DATASET}/test'
DATASET_ZIP = f'{DATASET}.zip'
DRIVE_DATASET_ZIP = f'{DRIVE}/dataset.zip'

def clear():
  if os.path.exists(DESKTOP):
    os.system('cls')
  else:
    clear_output()

# For all following folders,
# if the folders already exists delete the folder and make the folders and subfolders
print('Deleting old folders and making new empty folders...')
msg = 'Deleting old folders and making new empty folders...\n'
if os.path.exists(DATASET):
  shutil.rmtree(DATASET, onerror=lambda a,b,c:0)
for FOLDER in [DATASET, RENAMED_FILTERED_SAMPLES, RAW, NO_DIST, TRAIN, VALIDATE, TEST]:
  os.mkdir(FOLDER)
  if FOLDER not in [DATASET, RENAMED_FILTERED_SAMPLES]:
    for class_ in CLASSES:
      os.mkdir(f'{FOLDER}/{class_}')

# Copy the zip file into colab and extract it
print('Copying zip file from drive to colab file system...')
msg += 'Copying zip file from drive to colab file system...\n'
shutil.copy(src=DRIVE_RENAMED_FILTERED_SAMPLES_ZIP, dst=DATASET)
shutil.unpack_archive(filename=RENAMED_FILTERED_SAMPLES_ZIP,
                      extract_dir=RENAMED_FILTERED_SAMPLES, format='zip')
os.unlink(RENAMED_FILTERED_SAMPLES_ZIP)

# Pick random NUM_SAMPLES samples from the renamed filtered samples
print(f'Selecting random {NUM_SAMPLES} raw images...')
msg += f'Selecting random {NUM_SAMPLES} raw images...\n'
for class_ in CLASSES:
  files = random.sample(os.listdir(f'{RENAMED_FILTERED_SAMPLES}/{class_}'), k=NUM_SAMPLES)
  for file_num, file in enumerate(files):
    shutil.copy(src=f'{RENAMED_FILTERED_SAMPLES}/{class_}/{file}',
                dst=f'{RAW}/{class_}/{file_num:05}.jpg')

# Debugging
for class_ in CLASSES:
  print(class_, len(os.listdir(f'{RAW}/{class_}')))
  msg += f"{class_} {len(os.listdir(f'{RAW}/{class_}'))}\n"

# Apply 7 non distorting transformations to the raw images
# and save the original image as well in the no dist folder
print('Applying non distorting transformations on the raw images...')
msg += 'Applying non distorting transformations on the raw images...\n'
def non_dist(tid):
  class_ = CLASSES[tid]
  img_cntr = 0
  for file_num in range(NUM_SAMPLES):
    img = Image.open(f'{RAW}/{class_}/{file_num:05}.jpg')
    img.save(f'{NO_DIST}/{class_}/{img_cntr:05}.jpg')
    img_cntr += 1
    for op in [Image.ROTATE_90, Image.ROTATE_180, Image.ROTATE_270,
              Image.FLIP_LEFT_RIGHT, Image.FLIP_TOP_BOTTOM,
              Image.TRANSPOSE, Image.TRANSVERSE]:
      img.transpose(op).save(f'{NO_DIST}/{class_}/{img_cntr:05}.jpg')
      img_cntr += 1
      thread_msg[tid] = f'Processed {img_cntr} images'
  thread_msg[tid] = 'Thread completed'
  time.sleep(1)
  thread_finished[tid] = True
thread_msg = {tid: '' for tid in range(4)}
thread_finished = {tid: False for tid in range(4)}
for tid in range(4):
  threading.Thread(target=non_dist, args=(tid,)).start()
while not all(thread_finished.values()):
  clear()
  print(msg)
  print(*thread_msg.items(), sep='\n')
  time.sleep(1)

# Debugging
for class_ in CLASSES:
  print(class_, len(os.listdir(f'{NO_DIST}/{class_}')))
  msg += f"{class_} {len(os.listdir(f'{NO_DIST}/{class_}'))}\n"

# Applying distorting transformations to the non distorted images for training
print('Applying distorting transformations on the images...')
msg += 'Applying distorting transformations on the images...\n'
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
  rotation_range = 44,
  width_shift_range = 0.2,
  height_shift_range=0.2,
  shear_range=0.2,
  zoom_range=0.2,
  horizontal_flip=True,
  vertical_flip=True,
)
val_test_datagen = tf.keras.preprocessing.image.ImageDataGenerator()
def dist(tid):
  class_ = CLASSES[tid]
  num_files = NUM_SAMPLES * 8
  img_cntr = 0
  for file_num in range(int(TRAIN_SPLIT * num_files)):
    img = np.asarray(Image.open(f'{NO_DIST}/{class_}/{img_cntr:05}.jpg')) \
      .reshape(1, IMG_SIZE, IMG_SIZE, 3)
    for _ in train_datagen.flow(
      x = img,
      batch_size = 1,
      shuffle = False,
      save_to_dir = f'{TRAIN}/{class_}',
      save_prefix = f'{img_cntr:05}',
      save_format = 'jpeg'
    ):
      l = len(os.listdir(f'{TRAIN}/{class_}'))
      thread_msg[tid] = f'Processed {l} train images'
      if l == (file_num + 1) * KERAS_DISTORTION_SCALE:
        img_cntr += 1
        break
  for file_num in range(int((VALIDATE_SPLIT - TRAIN_SPLIT) * num_files)):
    img = np.asarray(Image.open(f'{NO_DIST}/{class_}/{img_cntr:05}.jpg')) \
      .reshape(1, IMG_SIZE, IMG_SIZE, 3)
    for _ in val_test_datagen.flow(
      x = img,
      batch_size = 1,
      shuffle = False,
      save_to_dir = f'{VALIDATE}/{class_}',
      save_prefix = f'{img_cntr:05}',
      save_format = 'jpeg'
    ):
      l = len(os.listdir(f'{VALIDATE}/{class_}'))
      thread_msg[tid] = f'Processed {l} validate images'
      if l == (file_num + 1):
        img_cntr += 1
        break
  for file_num in range(int((1 - VALIDATE_SPLIT) * num_files)):
    img = np.asarray(Image.open(f'{NO_DIST}/{class_}/{img_cntr:05}.jpg')) \
      .reshape(1, IMG_SIZE, IMG_SIZE, 3)
    for _ in val_test_datagen.flow(
      x = img,
      batch_size = 1,
      shuffle = False,
      save_to_dir = f'{TEST}/{class_}',
      save_prefix = f'{img_cntr:05}',
      save_format = 'jpeg'
    ):
      l = len(os.listdir(f'{TEST}/{class_}'))
      thread_msg[tid] = f'Processed {l} test images'
      if l == (file_num + 1):
        img_cntr += 1
        break
  thread_msg[tid] = 'Thread completed'
  time.sleep(1)
  thread_finished[tid] = True
thread_msg = {tid: '' for tid in range(4)}
thread_finished = {tid: False for tid in range(4)}
for tid in range(4):
  threading.Thread(target=dist, args=(tid,)).start()
while not all(thread_finished.values()):
  clear()
  print(msg)
  print(*thread_msg.items(), sep='\n')
  time.sleep(1)

# Debugging
for class_ in CLASSES:
  print(class_, len(os.listdir(f'{TRAIN}/{class_}')))
  print(class_, len(os.listdir(f'{VALIDATE}/{class_}')))
  print(class_, len(os.listdir(f'{TEST}/{class_}')))

# Archive the root and copy to drive
print('Archiving the root folder and storing it in drive...')
shutil.make_archive(DATASET, 'zip', DATASET)
if os.path.exists(DRIVE_DATASET_ZIP):
  os.unlink(DRIVE_DATASET_ZIP)
shutil.move(DATASET_ZIP, DRIVE)

# Note ending time
end = time.time()

# Calculate time difference
exec_time = int(end - start)

# Print execution time
print(f'Total execution time: {exec_time}s (' + \
      f'{exec_time // 3600}h {(exec_time // 60) % 60}m {exec_time % 60}s)')