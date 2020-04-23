import shutil
import os
import time
import tensorflow as tf
import numpy as np
from PIL import Image
CLASSES = 'cdnp'
FOLDER_PREFIX = 'F:/github/Drones_For_Structural_Audit/dataset/internal'
INPUT_FOLDER = f'{FOLDER_PREFIX}/256_raw'
OUTPUT_FOLDER = f'{FOLDER_PREFIX}/256_split'
AUGMENTATION_FOLDER = f'{FOLDER_PREFIX}/256_aug'
if os.path.exists(OUTPUT_FOLDER):
  shutil.rmtree(OUTPUT_FOLDER, onerror=lambda _: _)
if os.path.exists(AUGMENTATION_FOLDER):
  shutil.rmtree(AUGMENTATION_FOLDER, onerror=lambda _: _)
time.sleep(1)
os.mkdir(OUTPUT_FOLDER)
os.mkdir(AUGMENTATION_FOLDER)
for class_ in CLASSES:
  os.mkdir(f'{OUTPUT_FOLDER}/{class_}')
  os.mkdir(f'{AUGMENTATION_FOLDER}/{class_}')
num_samples = {class_: 0 for class_ in CLASSES}
with os.scandir(INPUT_FOLDER) as folder:
  for file in folder:
    label = int(file.name.split('_')[3][0])
    num_samples[CLASSES[label]] += 1
    shutil.copy(src=file.path, dst=f'{OUTPUT_FOLDER}/{CLASSES[label]}')
aug_ratio = {class_: 10_000 // num_samples[class_] for class_ in CLASSES}
data_gen = tf.keras.preprocessing.image.ImageDataGenerator(
  zca_whitening = True,
  rotation_range = 40,
  width_shift_range = 0.2,
  height_shift_range = 0.2,
  brightness_range = [0.8, 1.2],
  shear_range = 0.2,
  zoom_range = 0.2,
  horizontal_flip = True,
  vertical_flip = True,
)
for class_ in CLASSES:
  with os.scandir(f'{OUTPUT_FOLDER}/{class_}') as folder:
    for f, file in enumerate(folder, start=1):
      img = np.asarray(Image.open(file.path)).reshape((1, 256, 256, 3))
      for _ in data_gen.flow(
                  img,
                  save_to_dir=f'{AUGMENTATION_FOLDER}/{class_}',
                  batch_size=1,
                  shuffle=False,
                  save_format='jpeg',
                  save_prefix=f'{file.name[:-4]}'
                ):
        if len(os.listdir(f'{AUGMENTATION_FOLDER}/{class_}')) == aug_ratio[class_]*f:
          break