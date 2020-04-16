from PIL import Image
import numpy as np
import threading
import time
import pickle
import os

# Note start time
start = time.time()

# Final output dimensions
IMAGE_SIZE = 256

# Folder paths
FOLDER_PREFIX = 'F:\\github\\Drones_For_Structural_Audit\\dataset\\internal'
INPUT_FOLDER = f'{FOLDER_PREFIX}\\300'
OUTPUT_FOLDER = f'{FOLDER_PREFIX}\\256'

# All classes
CLASSES = 'cdnp'

# Function that the thread will run
def f(class_):

  # Class folders that this thread will work on
  INPUT_CLASS_FOLDER = f'{INPUT_FOLDER}\\{class_}'
  OUTPUT_CLASS_FOLDER = f'{OUTPUT_FOLDER}\\{class_}'

  # Set progress image
  thread_msg[class_] = 'Counting images...'

  # Initialize total number of images
  total_img_num = 0

  # Scan the folder and iterate through all files in it
  with os.scandir(INPUT_CLASS_FOLDER) as folder:
    for file in folder:
      
      # Open the image at the file's path
      img = Image.open(file.path)

      # Store the image dimensions
      w, h = img.width, img.height

      # Count image number
      total_img_num += (w - IMAGE_SIZE + 1) * (h - IMAGE_SIZE + 1)
  
  # Store numeric label value
  Y[class_] = np.tile(CLASSES.find(class_), total_img_num)

  # Number of digits in the file name of output images
  num_digits = len(str(total_img_num))

  # Counter for numbering images
  img_cntr = 0

  # Temporary list for storing X data
  x_temp = []

  # Scan the folder and iterate through all files in it
  with os.scandir(INPUT_CLASS_FOLDER) as folder:
    for file in folder:

      # Open the image at the file's path
      img = Image.open(file.path)

      # Store the image dimensions
      w, h = img.width, img.height

      # Iterate for all sliding windows of size IMAGE_SIZE * IMAGE_SIZE
      for y in range(h - IMAGE_SIZE + 1):
        for x in range(w - IMAGE_SIZE + 1):

          # Set progress msg
          thread_msg[class_] = f'Processing {img_cntr}/{total_img_num}...'

          # Crop the image at the window
          cropped_img = img.crop((x, y, x + IMAGE_SIZE, y + IMAGE_SIZE))

          # Store the pixels in the x_temp
          x_temp.append(np.asarray(cropped_img))

          # Save the image in the output folder
          cropped_img.save(f'{OUTPUT_CLASS_FOLDER}\\{img_cntr:0{num_digits}}.jpg')

          # Increment the image counter
          img_cntr += 1
  
  # Convert list to np array
  X[class_] = np.asarray(x_temp)

  # Set progress msg
  thread_msg[class_] = f'Thread finished'
  thread_finished[class_] = True


# Message and boolean for thread progress
thread_msg = {class_: 'Thread not started' for class_ in CLASSES}
thread_finished = {class_: False for class_ in CLASSES}

# These will store the numeric data
X, Y = dict(), dict()

# Start all threads
for class_ in CLASSES:
  threading.Thread(target=f, args=(class_,)).start()

# While all threads have not finished
while not all(thread_finished.values()):

  # Sleep for 1 second
  time.sleep(1)
  os.system('cls')
  print(*[f'{class_}: {thread_msg[class_]}' for class_ in CLASSES], sep='\n')

# Flatten all the X and Y values
X = np.asarray([X[class_] for class_ in CLASSES])
Y = np.asarray([Y[class_] for class_ in CLASSES])

# Print shapes of X and Y
print(f'X shape: {X.shape}')
print(f'Y shape: {Y.shape}')

# Pickle X and Y
with open(f'{OUTPUT_FOLDER}\\X', 'wb') as f:
  pickle.dump(X, f)
with open(f'{OUTPUT_FOLDER}\\Y', 'wb') as f:
  pickle.dump(Y, f)

# Note end time
end = time.time()

# Calculate execution time
exec_time = int(end - start)
h, m, s = (exec_time // 3600), (exec_time // 60) % 60, exec_time % 60

# Print execution time
print(f'Total time of execution {h} hours, {m} minutes, {s} seconds')