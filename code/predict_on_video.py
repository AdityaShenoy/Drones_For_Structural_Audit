from PIL import Image
import numpy as np
import time
import os
import shutil
import random

# Constants
IMG_SIZE = 256
COLOR_CODE = (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)

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
      
      prediction = random.randint(0, 3)

      window_result = Image.new('RGB', (IMG_SIZE, IMG_SIZE), 
        COLOR_CODE[prediction])
      result_img.paste(window_result, box=(w + x, y) if w < h else (x, h + y))
  result_img.save('C:/Users/admin/Desktop/test/test.jpg')

# TRIAL DRIVER CODE
with os.scandir(
  'F:/github/Drones_For_Structural_Audit/dataset/raw/all_images') as folder:
  for file in folder:
    img = Image.open(file.path)
    predict(img)
    input('DONE')