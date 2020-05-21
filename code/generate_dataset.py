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
TRAIN_SPLIT = 0.8
CLASSES = 'cdnp'
NUM_SAMPLES = 625
NUM_CHANNELS = 3

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
for FOLDER in [DATASET, RENAMED_FILTERED_SAMPLES, RAW, NO_DIST, TRAIN, VALIDATE]:
  os.mkdir(FOLDER)
  if FOLDER not in [DATASET, RENAMED_FILTERED_SAMPLES]:
    for class_ in CLASSES:
      os.mkdir(f'{FOLDER}/{class_}')

# Copy the zip file into colab and extract it
print('Copying and extracting zip file from drive to colab file system...')
msg += 'Copying and extracting zip file from drive to colab file system...\n'
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
val_datagen = tf.keras.preprocessing.image.ImageDataGenerator()
def dist(tid):
  class_ = CLASSES[tid]
  num_files = NUM_SAMPLES * 8
  img_cntr = 0
  for file_num in range(round(TRAIN_SPLIT * num_files)):
    img = np.asarray(Image.open(f'{NO_DIST}/{class_}/{img_cntr:05}.jpg')) \
      .reshape(1, IMG_SIZE, IMG_SIZE, NUM_CHANNELS)
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
  for file_num in range(round((1 - TRAIN_SPLIT) * num_files)):
    img = np.asarray(Image.open(f'{NO_DIST}/{class_}/{img_cntr:05}.jpg')) \
      .reshape(1, IMG_SIZE, IMG_SIZE, NUM_CHANNELS)
    for _ in val_datagen.flow(
      x = img,
      batch_size = 1,
      shuffle = False,
      save_to_dir = f'{VALIDATE}/{class_}',
      save_prefix = f'{img_cntr:05}',
      save_format = 'jpeg'
    ):
      thread_msg[tid] = f'Processed {file_num + 1} validate images'
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
  msg += f"{class_} {len(os.listdir(f'{TRAIN}/{class_}'))}\n"
  msg += f"{class_} {len(os.listdir(f'{VALIDATE}/{class_}'))}\n"
  print(class_, len(os.listdir(f'{TRAIN}/{class_}')))
  print(class_, len(os.listdir(f'{VALIDATE}/{class_}')))

# Generate binary dataset
msg += 'Generating dataset for binary classification...\n'
print('Generating dataset for binary classification...')
train_distribution = [2667, 2667, 2666]
validate_distribution = [334, 333, 333]
def bin_data(tid):
  thread_msg[tid] = 'Thread started'
  class_ = CLASSES[tid]
  os.mkdir(f'{DATASET}/bin_{class_}')
  os.mkdir(f'{DATASET}/bin_{class_}/train')
  os.mkdir(f'{DATASET}/bin_{class_}/train/not_{class_}')
  os.mkdir(f'{DATASET}/bin_{class_}/validate')
  os.mkdir(f'{DATASET}/bin_{class_}/validate/not_{class_}')
  shutil.copytree(src=f'{TRAIN}/{class_}',
                  dst=f'{DATASET}/bin_{class_}/train/{class_}')
  shutil.copytree(src=f'{VALIDATE}/{class_}',
                  dst=f'{DATASET}/bin_{class_}/validate/{class_}')
  file_num = 0
  for i, neg_class in enumerate(CLASSES.replace(class_, '')): # dnp for c, cnp for d, and so on
    files = random.sample(os.listdir(f'{TRAIN}/{neg_class}'),
                          k=train_distribution[i])
    for file in files:
      shutil.copy(src=f'{TRAIN}/{neg_class}/{file}',
                  dst=f'{DATASET}/bin_{class_}/train/not_{class_}/{file_num:05}.jpg')
      file_num += 1
      thread_msg[tid] = f'Copied {file_num} training images'
  thread_msg[tid] = 'Copying validating images'
  file_num = 0
  for i, neg_class in enumerate(CLASSES.replace(class_, '')): # dnp for c, cnp for d, and so on
    files = random.sample(os.listdir(f'{VALIDATE}/{neg_class}'),
                          k=validate_distribution[i])
    for file in files:
      shutil.copy(src=f'{VALIDATE}/{neg_class}/{file}',
                  dst=f'{DATASET}/bin_{class_}/validate/not_{class_}/{file_num:05}.jpg')
      file_num += 1
      thread_msg[tid] = f'Copied {file_num} validating images'
  thread_msg[tid] = 'Thread completed'
  thread_finished[tid] = True
NUM_THREADS = 4
thread_msg = {tid: 'Thread not started' for tid in range(NUM_THREADS)}
thread_finished = {tid: False for tid in range(NUM_THREADS)}
for tid in range(NUM_THREADS):
  threading.Thread(target=bin_data, args=(tid,)).start()
while not all(thread_finished.values()):
  clear()
  print(msg)
  print(*thread_msg.items(), sep='\n')
  time.sleep(1)
clear()
print(msg)

# Debugging
for class_ in CLASSES:
  print(f'not_{class_}', len(os.listdir(f'{DATASET}/bin_{class_}/train/{class_}')))
  print(f'not_{class_}', len(os.listdir(f'{DATASET}/bin_{class_}/train/not_{class_}')))
  print(f'not_{class_}', len(os.listdir(f'{DATASET}/bin_{class_}/validate/{class_}')))
  print(f'not_{class_}', len(os.listdir(f'{DATASET}/bin_{class_}/validate/not_{class_}')))

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