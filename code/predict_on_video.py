from PIL import Image
import numpy as np
import tensorflow as tf
import time
import os
import shutil
import random

# Constants
IMG_SIZE = 256
COLOR_CODE = (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0) #cdnp

# File and folder paths
DESKTOP = 'C:/Users/admin/Desktop/content'
CONTENT = '/content' if not os.path.exists(DESKTOP) else DESKTOP
DRIVE = f'{CONTENT}/drive/My Drive'
DRIVE_MODEL_ZIP = f'{DRIVE}/model.zip'
MODEL = f'{CONTENT}/model'
MODEL_ZIP = f'{MODEL}.zip'
SAVE = f'{MODEL}/save/'
DRIVE_ALL_IMAGES_ZIP = f'{DRIVE}/all_images.zip'
ALL_IMAGES_ZIP = f'{CONTENT}/all_images.zip'
ALL_IMAGES = f'{CONTENT}/all_images'
OUTPUT = f'{CONTENT}/output'
OUTPUT_ZIP = f'{OUTPUT}.zip'
DRIVE_OUTPUT_ZIP = f'{DRIVE}/output.zip'

# For all following folders,
# if the folders already exists delete the folder and make the folders and subfolders
print('Deleting old folders and making new empty folders...')
for FOLDER in [MODEL, ALL_IMAGES, OUTPUT]:
  if os.path.exists(FOLDER):
    shutil.rmtree(FOLDER, onerror=lambda a,b,c:0)
  os.mkdir(FOLDER)
for src, inter, dst in [(DRIVE_MODEL_ZIP, MODEL_ZIP, MODEL),
                        (DRIVE_ALL_IMAGES_ZIP, ALL_IMAGES_ZIP, ALL_IMAGES)]:
  shutil.copy(src, CONTENT)
  shutil.unpack_archive(inter, dst)
  os.unlink(inter)

# This function performs sliding window mechanism
# and predicts the class of each window
def predict(img):

  # Round down the dimensions of the image
  w, h = img.width // IMG_SIZE * IMG_SIZE, img.height // IMG_SIZE * IMG_SIZE
  img = img.resize((w, h))

  # Make a new result image
  result_img = Image.new(mode='RGB', size=(2 * w, h) if w < h else (w, 2 * h))
  result_img.paste(img, box=(0, 0))
  
  # Sliding window mechanism
  for y in range(0, h, IMG_SIZE):
    for x in range(0, w, IMG_SIZE):
      window = img.crop((x, y, x + IMG_SIZE, y + IMG_SIZE))
      
      model_output = model.predict(
        np.asarray(window).reshape(1, IMG_SIZE, IMG_SIZE, 3))
      
      prediction = model_output[0].argmax()

      window_result = Image.new('RGB', (IMG_SIZE, IMG_SIZE), 
        COLOR_CODE[prediction])
      result_img.paste(window_result, box=(w + x, y) if w < h else (x, h + y))
  return result_img

# Load the model
print('Loading the model...')
model = tf.keras.models.load_model(SAVE)

# For all camera images, predict on the sliding window mechanism
print('Predicting on camera images...')
with os.scandir(ALL_IMAGES) as folder:
  for file_num, file in enumerate(folder, start=1):
    img = Image.open(file.path)
    result_img = predict(img)
    result_img.save(f'{OUTPUT}/{file.name}')
    print(f'Processed {file_num} images...')

# Archive and store output in drive
print('Archiving output folder and storing it in drive...')
shutil.make_archive(OUTPUT, 'zip', OUTPUT)
if os.path.exists(DRIVE_OUTPUT_ZIP):
  os.unlink(DRIVE_OUTPUT_ZIP)
shutil.move(OUTPUT_ZIP, DRIVE)