from PIL import Image
import os
import shutil

# Image dimension
IMG_SIZE = 512

# Folder paths
INPUT_FOLDER = 'F:/github/Drones_For_Structural_Audit/dataset/internal/grid/input'
OUTPUT_FOLDER = f'F:/github/Drones_For_Structural_Audit/dataset/internal/grid/{IMG_SIZE}'

# Class labels
CLASSES = 'cdnp'

# Delete folder if already exists
if os.path.exists(OUTPUT_FOLDER):
  shutil.rmtree(OUTPUT_FOLDER, onerror=lambda a,b,c:_)

# Make the output folder and subfolders
os.mkdir(OUTPUT_FOLDER)
for class_ in CLASSES:
  os.mkdir(f'{OUTPUT_FOLDER}/{class_}')

# For all class input folders and files in it
for class_ in CLASSES:

  # Initialize image counter for all classes
  img_cntr = 0
  with os.scandir(f'{INPUT_FOLDER}/{class_}') as folder:
    for file in folder:

      # Open the image file
      img = Image.open(file.path)

      # Extract dimensions of the image
      w, h = img.width, img.height

      # Number of grids to be divided
      w_factor, h_factor = max(1, w // IMG_SIZE), max(1, h // IMG_SIZE)

      up_down = '_up' if (w < IMG_SIZE or h < IMG_SIZE) else ''

      # For all grids
      for y in range(h_factor):
        for x in range(w_factor):

          # Crop the image in the grid and save it in the output folder
          img.crop((x * w // w_factor, y * h // h_factor,
                    (x + 1) * w // w_factor, (y + 1) * h // h_factor)). \
                  resize((256, 256)). \
                  save(f'{OUTPUT_FOLDER}/{class_}/{img_cntr:04}{up_down}.jpg')
          img_cntr += 1

  print(class_, img_cntr)