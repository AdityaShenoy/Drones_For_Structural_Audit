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
TRAIN_PROP = 0.8
CLASSES = 'cdnp'

# Folder and file paths
CONTENT = '/content'
DRIVE = f'{CONTENT}/drive/My Drive'
DRIVE_RENAMED_FILTERED_SAMPLES_ZIP = f'{DRIVE}/renamed_filtered_samples.zip'
ROOT = f'{CONTENT}/root'
RENAMED_FILTERED_SAMPLES_ZIP = f'{ROOT}/renamed_filtered_samples.zip'
RENAMED_FILTERED_SAMPLES = f'{ROOT}/renamed_filtered_samples'
RAW = f'{ROOT}/raw'
NO_DIST = f'{ROOT}/no_dist'
DIST = f'{ROOT}/dist'
TRAIN = f'{ROOT}/train'
TEST = f'{ROOT}/test'
ROOT_ZIP = f'{CONTENT}/root.zip'

# For all following folders,
# if the folders already exists delete the folder and make the folders and subfolders
print('Deleting old folders and making new empty folders...')
msg = 'Deleting old folders and making new empty folders...\n'
if os.path.exists(ROOT):
  shutil.rmtree(ROOT, onerror=lambda a,b,c:0)
for FOLDER in [ROOT, RENAMED_FILTERED_SAMPLES, RAW, NO_DIST, DIST, TRAIN, TEST]:
  os.mkdir(FOLDER)
  if FOLDER not in [ROOT, RENAMED_FILTERED_SAMPLES]:
    for class_ in CLASSES:
      os.mkdir(f'{FOLDER}/{class_}')

# Copy the zip file into colab and extract it
print('Copying zip file from drive to colab file system...')
msg += 'Copying zip file from drive to colab file system...\n'
shutil.copy(src=DRIVE_RENAMED_FILTERED_SAMPLES_ZIP, dst=ROOT)
shutil.unpack_archive(filename=RENAMED_FILTERED_SAMPLES_ZIP,
                      extract_dir=RENAMED_FILTERED_SAMPLES, format='zip')
os.unlink(RENAMED_FILTERED_SAMPLES_ZIP)

# Pick random 625 samples from the renamed filtered samples
print('Selecting random 625 raw images...')
msg += 'Selecting random 625 raw images...\n'
for class_ in CLASSES:
  files = random.sample(os.listdir(f'{RENAMED_FILTERED_SAMPLES}/{class_}'), k=625)
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
  with os.scandir(f'{RAW}/{class_}') as folder:
    for file in folder:
      img = Image.open(file.path)
      img.save(f'{NO_DIST}/{class_}/{img_cntr:05}.jpg')
      img_cntr += 1
      for op in [Image.ROTATE_90, Image.ROTATE_180, Image.ROTATE_270,
                Image.FLIP_LEFT_RIGHT, Image.FLIP_TOP_BOTTOM,
                Image.TRANSPOSE, Image.TRANSVERSE]:
        img.transpose(op).save(f'{NO_DIST}/{class_}/{img_cntr:05}.jpg')
        img_cntr += 1
        thread_msg[tid] = f'Processed {img_cntr} images'
  thread_msg[tid] = 'Thread completed'
  thread_finished[tid] = True
thread_msg = {tid: '' for tid in range(4)}
thread_finished = {tid: False for tid in range(4)}
for tid in range(4):
  threading.Thread(target=non_dist, args=(tid,)).start()
while not all(thread_finished.values()):
  clear_output()
  print(msg)
  print(*thread_msg.items(), sep='\n')
  time.sleep(1)

# Debugging
for class_ in CLASSES:
  print(class_, len(os.listdir(f'{NO_DIST}/{class_}')))
  msg += f"{class_} {len(os.listdir(f'{NO_DIST}/{class_}'))}\n"

# Applying distorting transformations to the non distorted images
print('Applying distorting transformations on the images...')
msg += 'Applying distorting transformations on the images...\n'
datagen = tf.keras.preprocessing.image.ImageDataGenerator(
  rotation_range = 44,
  width_shift_range = 0.2,
  height_shift_range=0.2,
  shear_range=0.2,
  zoom_range=0.2,
  horizontal_flip=True,
  vertical_flip=True,
)
def dist(tid):
  class_ = CLASSES[tid]
  with os.scandir(f'{NO_DIST}/{class_}') as folder:
    for file_num, file in enumerate(folder, start=1):
      img = np.asarray(Image.open(file.path)).reshape(1, IMG_SIZE, IMG_SIZE, 3)
      for _ in datagen.flow(
        x = img,
        batch_size = 1,
        shuffle = False,
        save_to_dir = f'{DIST}/{class_}',
        save_prefix = file.name[:-4],
        save_format = 'jpeg'
      ):
        l = len(os.listdir(f'{DIST}/{class_}'))
        thread_msg[tid] = f'Processed {l} images'
        if l == file_num * KERAS_DISTORTION_SCALE:
          break
  thread_msg[tid] = 'Thread completed'
  thread_finished[tid] = True
thread_msg = {tid: '' for tid in range(4)}
thread_finished = {tid: False for tid in range(4)}
for tid in range(4):
  threading.Thread(target=dist, args=(tid,)).start()
while not all(thread_finished.values()):
  clear_output()
  print(msg)
  print(*thread_msg.items(), sep='\n')
  time.sleep(1)

# Debugging
for class_ in CLASSES:
  print(class_, len(os.listdir(f'{DIST}/{class_}')))
  msg += f"{class_} {len(os.listdir(f'{DIST}/{class_}'))}\n"

# Split the distorted images into train and test folders
print('Splitting the transformed images into train and test folders...')
msg += 'Splitting the transformed images into train and test folders...'
def split(tid):
  class_ = CLASSES[tid]
  files = os.listdir(f'{DIST}/{class_}')
  thread_msg[tid] = 'Copying training images'
  for file_num, file in enumerate(files[:int(TRAIN_PROP * len(files))]):
    shutil.copy(src=f'{DIST}/{class_}/{file}',
                dst=f'{TRAIN}/{class_}/{file_num:05}.jpg')
  thread_msg[tid] = 'Copying testing images'
  for file_num, file in enumerate(files[int(TRAIN_PROP * len(files)):]):
    shutil.copy(src=f'{DIST}/{class_}/{file}',
                dst=f'{TEST}/{class_}/{file_num:05}.jpg')
  thread_msg[tid] = 'Thread completed'
  thread_finished[tid] = True
thread_msg = {tid: '' for tid in range(4)}
thread_finished = {tid: False for tid in range(4)}
for tid in range(4):
  threading.Thread(target=split, args=(tid,)).start()
while not all(thread_finished.values()):
  clear_output()
  print(msg)
  print(*thread_msg.items(), sep='\n')
  time.sleep(1)
clear_output()
print(msg)

# Debugging
for class_ in CLASSES:
  print(class_, len(os.listdir(f'{TRAIN}/{class_}')))
  print(class_, len(os.listdir(f'{TEST}/{class_}')))

# Archive the root and copy to drive
shutil.make_archive(ROOT_ZIP[:-4], 'zip', ROOT)
shutil.move(ROOT_ZIP, DRIVE)

# Note ending time
end = time.time()

# Calculate time difference
exec_time = int(end - start)

# Print execution time
print(f'Total execution time: {exec_time}s (' + \
      f'{exec_time // 3600}h {(exec_time // 60) % 60}m {exec_time % 60}s)')