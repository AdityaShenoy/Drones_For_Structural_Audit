from PIL import Image
import numpy as np
import tensorflow as tf
import time
import os

# Factor by which a single image is augmented with distortion
KERAS_AUG_FACTOR = 0

# Training proportion
TRAIN_PROP = 0.8

# Dataset size
# Training and testing size
DATASET_SIZE = 20_000
TRAINING_SIZE = int(TRAIN_PROP * DATASET_SIZE)
TESTING_SIZE = DATASET_SIZE - TRAINING_SIZE

# Batch size
BATCH_SIZE = 32

# Image size
IMG_SIZE = 256

# Folder paths
DATASET_FOLDER = 'F:/github/Drones_For_Structural_Audit/dataset/internal/grid/no_dist'

# Class labels
CLASSES = 'cdnp'

# Generate augmented images based on train or test mode
def generator(mode):

  # Generator functions need to run infinitely
  while True:

    # Initialize empty lists
    X, Y = [], []

    # For all classes
    for label, class_ in enumerate(CLASSES):

      # Extract all file names
      files = os.listdir(f'{DATASET_FOLDER}/{class_}')

      # Split the dataset based on mode
      if mode == 'train':
        files = files[:int(0.8 * len(files))]
      elif mode == 'test':
        files = files[int(0.8 * len(files)):]
      
      # For all files
      for file in files:

        # Open the image file
        img = Image.open(f'{DATASET_FOLDER}/{class_}/{file}')

        # Extract pixels of the image
        pixels = np.asarray(img)

        # If keras augmentation factor is non zero
        # Augmentation is required only while training
        if KERAS_AUG_FACTOR and mode == 'train':

          # Keras data generator requires rank 4
          pixels = pixels.reshape((1, IMG_SIZE, IMG_SIZE, 3))

          # Augment the pixels of the image
          for aug_img in data_gen.flow(x=pixels, batch_size=1, shuffle=False):

            # Append the augmented image and label to the result lists
            X.append(aug_img)
            Y.append(label)

            # If the count of samples has reached batch size, yield it
            if len(X) == BATCH_SIZE:
              yield np.asarray(X).reshape(BATCH_SIZE, IMG_SIZE, IMG_SIZE, 3), np.asarray(Y)

              # Reset the data
              X, Y = [], []
        
        # If keras augmentation factor is 0 it means use original images as it is
        # While testing no augmentation is required
        else:

          # Reshape the pixels to rank 3
          pixels = pixels.reshape((IMG_SIZE, IMG_SIZE, 3))

          # Append the image and label to the result lists
          X.append(pixels)
          Y.append(label)

          # If the count of samples has reached batch size, yield it
          if len(X) == BATCH_SIZE:
            yield np.asarray(X), np.asarray(Y)

            # Reset the data
            X, Y = [], []

# Keras data generator which performs distorting
# augmentations as specified in the parameter
data_gen = tf.keras.preprocessing.image.ImageDataGenerator(
  zca_whitening = True,
  rotation_range = 44,
  width_shift_range = 0.2,
  height_shift_range = 0.2,
  brightness_range = [0.8, 1.2],
  shear_range = 0.2,
  zoom_range = 0.2,
)

# Initialize a sequential model
model = tf.keras.models.Sequential([
  tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3), activation='relu', 
                         input_shape=(IMG_SIZE, IMG_SIZE, 3)),
  tf.keras.layers.MaxPooling2D(pool_size=2, strides=2),
  tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu'),
  tf.keras.layers.MaxPooling2D(pool_size=2, strides=2),
  tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
  tf.keras.layers.MaxPooling2D(pool_size=2, strides=2),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(128, activation='relu'),
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
  x = generator(mode='train'),
  epochs = 20,
  verbose = 1,
  steps_per_epoch = TRAINING_SIZE // BATCH_SIZE
)

# Evaluate the model
model.evaluate(
  x = generator(mode='test'),
  verbose = 1,
  steps = TESTING_SIZE // BATCH_SIZE
)