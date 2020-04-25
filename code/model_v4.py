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
FOLDER_PREFIX = f'F:/github/Drones_For_Structural_Audit/dataset/internal/{IMG_SIZE}'

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


# Initialize a sequential model
model = tf.keras.models.Sequential([
  tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3), activation='relu', 
                        input_shape=(IMG_SIZE, IMG_SIZE, 3)),
  tf.keras.layers.MaxPooling2D(pool_size=(3, 3)),
  tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3), activation='relu'),
  tf.keras.layers.MaxPooling2D(pool_size=(3, 3)),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(32, activation='relu'),
  tf.keras.layers.Dense(len(CLASSES), activation='softmax')
])

# Compile the model
model.compile(
  loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
  optimizer = 'adam',
  metrics = ['accuracy']
)

# Train the model
model.fit(
  x = gen(mode='train'),
  epochs = 1,
  verbose = 1,
  steps_per_epoch = TRAINING_SIZE // BATCH_SIZE
)

model.evaluate(
  x = gen(mode='test'),
  steps = TESTING_SIZE // BATCH_SIZE,
  verbose = 1
)

# Note ending time
end = time.time()

# Calculate time difference
exec_time = int(end - start)

# Print execution time
print(f'Total execution time: {exec_time}s (' + \
      f'{exec_time // 3600}h {(exec_time // 60) % 60}m {exec_time % 60}s)')