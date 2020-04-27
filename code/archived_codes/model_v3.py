import tensorflow as tf
import numpy as np
import time
import os
import shutil
from PIL import Image

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
FOLDER_PREFIX = f'/content/{IMG_SIZE}'

# If the folder already exists in colab file directory, delete it
if os.path.exists(FOLDER_PREFIX):
  shutil.rmtree(FOLDER_PREFIX, onerror=lambda a,b,c:_)
  time.sleep(1)

# Make the folder
os.mkdir(FOLDER_PREFIX)

# Unpack the archive saved in drive
shutil.unpack_archive('/content/drive/My Drive/256.zip', FOLDER_PREFIX)

# Generate batch size of samples (X, Y) based on mode (train or test)
def gen(mode):

  # Folder to retrieve images from based on train or test mode
  FOLDER = f'{FOLDER_PREFIX}/{mode}'

  # Initialize current batch to empty list
  X, Y = [], []

  # Generator should run infinitely
  while True:

    # Iterate for all classes
    for label, class_ in enumerate(CLASSES):

      # For all files in the class folder
      with os.scandir(f'{FOLDER}/{class_}') as folder:
        for file in folder:

          # Open the image
          img = Image.open(file.path)

          # Append the pixel data
          X.append(np.asarray(img))
          
          # Append the label number 0123 for cdnp present in file name
          Y.append(label)
          
          # If current batch is full
          if len(Y) == BATCH_SIZE:

            # Yield the current batch
            yield np.asarray(X), np.asarray(Y)

            # Reset the batch again
            X, Y = [], []


# Open the output folder
with open(f'{FOLDER_PREFIX}/model_summary.txt', 'w') as f:

  # Print and store the progress
  def progress(msg):
    f.write(f'{msg}\n')
    print(msg)
    
  # Architecture of the neural network
  hidden_nodes = [1]

  # While the hidden layers are less than the 
  while len(hidden_nodes) <= MAX_HIDDEN_LAYERS:

    # Print progress
    progress('='*100)
    progress('Current NN architecture:')
    progress(' → '.join(map(str, hidden_nodes)))

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
    history = model.fit(
      x = gen(mode='train'),
      epochs = 1,
      verbose = 1,
      steps_per_epoch = TRAINING_SIZE // BATCH_SIZE
    )

    # Evaluate the model
    metrics = model.evaluate(
      x = gen(mode='test'),
      verbose = 1,
      steps = TESTING_SIZE // BATCH_SIZE
    )

    # Print progress
    progress(history.history)
    progress(metrics)
    progress('='*100)

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
    
  # Archive the IMG_SIZE folder and copy it to drive
  shutil.make_archive(f'{IMG_SIZE}', 'zip', f'{IMG_SIZE}')
  shutil.copy(src=f'/content/{IMG_SIZE}.zip', dst='/content/drive/My Drive')

  # Note ending time
  end = time.time()

  # Calculate time difference
  exec_time = int(end - start)

  # Print execution time
  print(f'Total execution time: {exec_time}s (' + \
        f'{exec_time // 3600}h {(exec_time // 60) % 60}m {exec_time % 60}s)')