import tensorflow as tf
import numpy as np
import time
import os
import shutil
from PIL import Image
from IPython.display import clear_output
import sys

# Note start time
start = time.time()

# Constants
TRAINING_SIZE = 64_000
TESTING_SIZE = 16_000
BATCH_SIZE = 32
IMG_SIZE = 256
CLASSES = 'cdnp'
MAX_HIDDEN_LAYERS = 2
MAX_NODES_PER_LAYER = 256

# Folder paths
# FOLDER_PREFIX = f'F:/github/Drones_For_Structural_Audit/dataset/internal/{IMG_SIZE}'
FOLDER_PREFIX = f'/content/{IMG_SIZE}'

# Generate batch size of variable (X or Y) data based on mode (train or test)
def gen(var, mode):

  # Folder to retrieve images from based on train or test mode
  FOLDER = f'{FOLDER_PREFIX}/{mode}'

  # Initialize current batch to empty list
  res = []

  # Generator should run infinitely
  while True:

    # Iterate for all classes
    for class_ in CLASSES:

      # For all files in the class folder
      with os.scandir(f'{FOLDER}/{class_}') as folder:
        for file in folder:

          # If variable is X
          if var == 'x':
          
            # Open the image
            img = Image.open(file.path)

            # Append the pixel data
            res.append(np.asarray(img))
          
          # If variable is y
          else:

            # Append the label number 0123 for cdnp present in file name
            res.append(int(file.name.split('_')[3]))
          
          # If current batch is full
          if len(res) == BATCH_SIZE:

            # Yield the current batch
            yield np.asarray(res)

            # Reset the batch again
            res = []

# Set the output for printing to a file
sys.stdout = open(f'{FOLDER_PREFIX}/model_summary.txt', 'w')

# Architecture of the neural network
hidden_nodes = [1]

# While the hidden layers are less than the 
while len(hidden_nodes) <= MAX_HIDDEN_LAYERS:

  # Print progress
  print('='*100)
  print('Current NN architecture:')
  print(*hidden_nodes, sep=' → ')

  # Initialize a sequential model
  model = tf.keras.models.Sequential()

  # Add flatten layer at the input
  model.add(tf.keras.layers.Flatten(input_shape=(IMG_SIZE, IMG_SIZE, 3)))

  # Add hidden layers
  for h in hidden_nodes:
    model.add(tf.keras.layers.Dense(h, activation='relu'))
  
  # Add output layer
  model.add(tf.keras.layers.Dense(len(CLASSES)))

  # Compile the model
  model.compile(
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer = 'adam',
    metrics = ['accuracy']
  )

  # Train the model
  model.fit(
    x = gen(var='x', mode='train'),
    y = gen(var='y', mode='train'),
    epochs = 20,
    verbose = 2,
    steps_per_epoch = TRAINING_SIZE // BATCH_SIZE
  )

  # Evaluate the model
  model.evaluate(
    x = gen(var='x', mode='test'),
    y = gen(var='y', mode='test'),
    verbose = 2,
    steps_per_epoch = TESTING_SIZE // BATCH_SIZE
  )

  # Reconfigure the architecture
  if hidden_nodes[-1] == MAX_NODES_PER_LAYER:
    for i in range(-1, -len(hidden_nodes)-1, -1):
      if hidden_nodes[i] == MAX_NODES_PER_LAYER:
        hidden_nodes[i] = 1
      else:
        hidden_nodes[i] *= 2
        break
    else:
      hidden_nodes.append(1)
  else:
    hidden_nodes[-1] *= 2
  
  # Print progress
  print('='*100)

# Note ending time
end = time.time()

# Calculate time difference
exec_time = int(end - start)

# Print execution time
print(f'Total execution time: {exec_time}s (' + \
      f'{exec_time // 3600}h {(exec_time // 60) % 60}m {exec_time % 60}s)')