from os import scandir
from PIL import Image
import numpy as np
import tensorflow as tf

# Output image dimension
IMAGE_SIZE = 256

# Default batch size
BATCH_SIZE = 32

# The classes
CLASSES = 'cdnp'

# Input folder path
# INPUT_FOLDER = 'F:/github/Drones_For_Structural_Audit/dataset/internal/300_aug'
INPUT_FOLDER = '/content/drive/My Drive/Colab Notebooks/300_aug'

# This function calculates the number of training samples available
def calculate_data_size(mode):

  # Initialize result
  result = 0

  # Input folder path
  FOLDER = f'{INPUT_FOLDER}/{mode}'

  # Iterate for all classes
  for class_ in CLASSES:

    # Scan the class folder
    with scandir(f'{FOLDER}/{class_}') as folder:
      for file in folder:

        # Open the image in the path
        img = Image.open(file.path)

        # Extract dimensions of the image
        w, h = img.width, img.height

        # Add sub image count to the result
        result += (w - IMAGE_SIZE + 1) * (h - IMAGE_SIZE + 1)

  # Return all the count
  return result


# Calculate training and testing samples
NUM_TRAINING_SAMPLES = calculate_data_size(mode='train')
NUM_TESTING_SAMPLES = calculate_data_size(mode='test')

print(NUM_TRAINING_SAMPLES, 4*16*8*2025)
print(NUM_TESTING_SAMPLES, 4*4*8*2025)


def batch_generator(batch_size=BATCH_SIZE, mode='train'):

  # Input folder path
  FOLDER = f'{INPUT_FOLDER}/{mode}'

  # Initializing empty lists that will contain batches of samples
  X, Y = [], []

  # Initialize sample number of this batch
  sample_size = 0

  # Run the generator function indefinitely
  while True:

    # Iterate for all classes
    for label, class_ in enumerate(CLASSES):

      # Scan the class folder
      with scandir(f'{FOLDER}/{class_}') as folder:
        for file in folder:

          # Open the image in the path
          img = Image.open(file.path)

          # Extract dimensions of the image
          w, h = img.width, img.height

          # Iterate for all windows of the image
          for y in range(h - IMAGE_SIZE + 1):
            for x in range(w - IMAGE_SIZE + 1):

              # Crop the image in the window
              cropped_img = img.crop((x, y, x + IMAGE_SIZE, y + IMAGE_SIZE))

              # Append the result data
              X.append(np.asarray(cropped_img))
              Y.append(label)

              # Increment the sample size
              sample_size += 1

              # If the batch is full
              if sample_size == batch_size:

                # Convert the lists to ndarrays and normalize the input
                X = np.asarray(X) / 256
                Y = np.asarray(Y)

                # Yield the results
                yield X, Y

                # Reset the variables
                X, Y, sample_size = [], [], 0


# Previous trials with training and testing metrics

# no augmentation
# loss: 8.7520 - accuracy: 0.2500
# loss: 8.7482 - accuracy: 0.2501
# model = tf.keras.models.Sequential([
#   tf.keras.layers.Flatten(input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3)),
#   tf.keras.layers.Dense(128),
#   tf.keras.layers.Dense(len(CLASSES))
# ])
# model.compile(
#   loss = 'sparse_categorical_crossentropy',
#   optimizer = 'sgd',
#   metrics = ['accuracy']
# )

# no augmentation
# loss: 11.5463 - accuracy: 0.9956
# loss: 5070.3223 - accuracy: 0.2496
# model = tf.keras.models.Sequential([
#   tf.keras.layers.Flatten(input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3)),
#   tf.keras.layers.Dense(128, activation='relu'),
#   tf.keras.layers.Dropout(0.2),
#   tf.keras.layers.Dense(len(CLASSES))
# ])
# model.compile(
#   loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#   optimizer = 'adam',
#   metrics = ['accuracy']
# )

# augmented data
# 5577s 172ms/step - loss: 5.0860 - accuracy: 0.9929
# 465s 57ms/step - loss: 794.8063 - accuracy: 0.2500
# model = tf.keras.models.Sequential([
#   tf.keras.layers.Flatten(input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3)),
#   tf.keras.layers.Dense(128, activation='relu'),
#   tf.keras.layers.Dropout(0.2),
#   tf.keras.layers.Dense(len(CLASSES))
# ])
# model.compile(
#   loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#   optimizer = 'adam',
#   metrics = ['accuracy']
# )

# augmented data
# 2701s 83ms/step - loss: 1.2346 - accuracy: 0.8177
# 281s 35ms/step - loss: 5.1768 - accuracy: 0.2500
# model = tf.keras.models.Sequential([
#   tf.keras.layers.Flatten(input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3)),
#   tf.keras.layers.Dense(32, activation='relu'),
#   tf.keras.layers.Dropout(0.2),
#   tf.keras.layers.Dense(len(CLASSES))
# ])
# model.compile(
#   loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#   optimizer = 'adam',
#   metrics = ['accuracy']
# )

# augmented data
# 2137s 66ms/step - loss: 0.3637 - accuracy: 0.9223
# 281s 35ms/step - loss: 10.8132 - accuracy: 0.2500
# model = tf.keras.models.Sequential([
#   tf.keras.layers.Flatten(input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3)),
#   tf.keras.layers.Dense(16, activation='relu'),
#   tf.keras.layers.Dense(16, activation='relu'),
#   tf.keras.layers.Dropout(0.2),
#   tf.keras.layers.Dense(len(CLASSES))
# ])

# # Compile the model
# model.compile(
#   loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#   optimizer = 'adam',
#   metrics = ['accuracy']
# )

# augmented data, train test flipped
# 521s 64ms/step - loss: 0.4491 - accuracy: 0.9361
# 988s 30ms/step - loss: 5.7910 - accuracy: 0.2500
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3)),
  tf.keras.layers.Dense(16, activation='relu'),
  tf.keras.layers.Dense(16, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(len(CLASSES))
])

# Compile the model
model.compile(
  loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
  optimizer = 'adam',
  metrics = ['accuracy']
)

# Train the model
history = model.fit_generator(
  generator = batch_generator(),
  steps_per_epoch = int(NUM_TRAINING_SAMPLES / BATCH_SIZE),
  epochs = 1,
  verbose = 1
)

# Evaluate the model
metrics = model.evaluate_generator(
  generator = batch_generator(mode='test'),
  steps = int(NUM_TESTING_SAMPLES / BATCH_SIZE),
  verbose = 1
)