from os import system, scandir
from PIL import Image
import pickle
import numpy as np
import time

start = time.time()

CLASSES = 'cdnp'

IMAGE_SIZE = 256

FOLDER_PREFIX = 'F:\\github\\Drones_For_Structural_Audit\\dataset\\internal'
INPUT_FOLDER = f'{FOLDER_PREFIX}\\300'
OUTPUT_FOLDER = f'{FOLDER_PREFIX}\\{IMAGE_SIZE}'

X, Y = [], []

for label, class_ in enumerate(CLASSES):

  total_img = 0

  with scandir(f'{INPUT_FOLDER}\\{class_}') as folder:
    for file in folder:

      img = Image.open(file.path)

      w, h = img.width, img.height

      total_img += (w - IMAGE_SIZE + 1) * (h - IMAGE_SIZE + 1)
  
  img_cntr = 0

  Y += [label for _ in range(total_img)]

  with scandir(f'{INPUT_FOLDER}\\{class_}') as folder:
    for file in folder:

      img = Image.open(file.path)

      w, h = img.width, img.height

      for y in range(h - IMAGE_SIZE + 1):
        for x in range(w - IMAGE_SIZE + 1):

          system('cls')
          print(f'{class_}: {img_cntr} / {total_img}')

          cropped_img = img.crop((x, y, x + IMAGE_SIZE, y + IMAGE_SIZE))

          X.append(np.asarray(cropped_img))

          img_cntr += 1

X = np.asarray(X)
Y = np.asarray(Y)

print(X.shape)
print(Y.shape)

with open(f'{OUTPUT_FOLDER}\\X', 'wb') as f:
  pickle.dump(X)

with open(f'{OUTPUT_FOLDER}\\Y', 'wb') as f:
  pickle.dump(Y)

end = time.time()

exec_time = end - start

print(f'{exec_time}s')