import os
import shutil
import tensorflow as tf
import numpy as np
from PIL import Image
import random
import time
import matplotlib.pyplot as plt
import threading

# Note starting time
start = time.time()

# Constants
IMG_SIZE = 256
DATASET_SIZE = 20_000
TRAIN_PROP = 0.8
TRAINING_SIZE = int(DATASET_SIZE * TRAIN_PROP)
TESTING_SIZE = DATASET_SIZE - TRAINING_SIZE
BATCH_SIZE = 32
CLASSES = 'cdnp'

# Folder and file paths
RENAMED_FILTERED_SAMPLES_DRIVE_ZIP = '/content/drive/My Drive/renamed_filtered_samples.zip'
RENAMED_FILTERED_SAMPLES_ZIP = '/content/renamed_filtered_samples.zip'
RENAMED_FILTERED_SAMPLES = '/content/renamed_filtered_samples'
RAW = '/content/raw'
NO_DIST = '/content/no_dist'
DIST = '/content/dist'
TRAIN = '/content/train'
TEST = '/content/test'
PLOTS = '/content/plots'

# For all following folders,
# if the folders already exists delete the folder and make the folders and subfolders
print('Deleting old folders and making new empty folders...')
for FOLDER in [RENAMED_FILTERED_SAMPLES, RAW, NO_DIST, DIST, TRAIN, TEST, PLOTS]:
  if os.path.exists(FOLDER):
    shutil.rmtree(FOLDER, onerror=lambda a,b,c:0)
  os.mkdir(FOLDER)
  if FOLDER not in [RENAMED_FILTERED_SAMPLES, PLOTS]:
    for class_ in CLASSES:
      os.mkdir(f'{FOLDER}/{class_}')

# Copy the zip file into colab and extract it
print('Deleting old zip file and copying zip file from drive to colab file system...')
if os.path.exists(RENAMED_FILTERED_SAMPLES_ZIP):
  os.unlink(RENAMED_FILTERED_SAMPLES_ZIP)
shutil.copy(src=RENAMED_FILTERED_SAMPLES_DRIVE_ZIP, dst='/content')
shutil.unpack_archive(filename=RENAMED_FILTERED_SAMPLES_ZIP,
                      extract_dir=RENAMED_FILTERED_SAMPLES, format='zip')

# Pick random 625 samples from the renamed filtered samples
print('Selecting random 625 raw images...')
for class_ in CLASSES:
  files = random.sample(os.listdir(f'{RENAMED_FILTERED_SAMPLES}/{class_}'), k=625)
  for file in files:
    shutil.copy(src=f'{RENAMED_FILTERED_SAMPLES}/{class_}/{file}',
                dst=f'{RAW}/{class_}')

# Apply 7 non distorting transformations to the raw images
# and save the original image as well in the no dist folder
print('Applying non distorting transformations on the raw images...')
for class_ in CLASSES:
  img_cntr = 0
  with os.scandir(f'{RAW}/{class_}') as folder:
    for file in folder:
      img = Image.open(file.path)
      img.save(f'{NO_DIST}/{class_}/{img_cntr:04}.jpg')
      img_cntr += 1
      for op in [Image.ROTATE_90, Image.ROTATE_180, Image.ROTATE_270,
                Image.FLIP_LEFT_RIGHT, Image.FLIP_TOP_BOTTOM,
                Image.TRANSPOSE, Image.TRANSVERSE]:
        img.transpose(op).save(f'{NO_DIST}/{class_}/{img_cntr:04}.jpg')
        img_cntr += 1

# Applying distorting transformations to the non distorted images
print('Applying distorting transformations on the images...')


# Split the non distorted images into train and test folders
print('Splitting the transformed images into train and test folders...')
for class_ in CLASSES:
  files = os.listdir(f'{NO_DIST}/{class_}')
  for file in files[:int(TRAIN_PROP * len(files))]:
    shutil.copy(src=f'{NO_DIST}/{class_}/{file}',
                dst=f'{TRAIN}/{class_}')
  for file_num, file in enumerate(files[int(TRAIN_PROP * len(files)):]):
    shutil.copy(src=f'{NO_DIST}/{class_}/{file}',
                dst=f'{TEST}/{class_}/{file_num:04}.jpg')

# Initialize the ML model
print('Building model...')
model = tf.keras.models.Sequential([
  tf.keras.layers.Conv2D(filters=16, kernel_size=(3,3), activation='relu',
                         input_shape=(IMG_SIZE, IMG_SIZE, 3)),
  tf.keras.layers.MaxPooling2D(2, 2),
  tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3), activation='relu'),
  tf.keras.layers.MaxPooling2D(2, 2),
  tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), activation='relu'),
  tf.keras.layers.MaxPooling2D(2, 2),
  tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), activation='relu'),
  tf.keras.layers.MaxPooling2D(2, 2),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(512, activation='relu'),
  tf.keras.layers.Dense(len(CLASSES), activation='softmax')
])

# Visualize the model
tf.keras.utils.plot_model(model, to_file=f'{PLOTS}/model.jpg', show_shapes=True)

# Complile the model
print('Compiling the mdoel...')
model.compile(
  loss = 'categorical_crossentropy',
  optimizer = tf.keras.optimizers.Adam(lr=0.001),
  metrics = ['accuracy']
)

# Set up the training and testing dataset generator
print('Setting up dataset generators...')
train_gen = tf.keras.preprocessing.image.ImageDataGenerator(
  rescale = 1/255,
  rotation_range = 40,
  width_shift_range = 0.2,
  height_shift_range = 0.2,
  shear_range = 0.2,
  zoom_range = 0.2,
  horizontal_flip = True,
  vertical_flip = True,
).flow_from_directory(directory = TRAIN)
test_gen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1/255)\
                 .flow_from_directory(directory = TEST)

# Training the model
print('Training the model...')
history = model.fit(
  x = train_gen,
  epochs = 15,
  verbose = 1,
  validation_data = test_gen,
  steps_per_epoch = TRAINING_SIZE // BATCH_SIZE,
  validation_steps = TESTING_SIZE // BATCH_SIZE
)

# Values for the graphs
print('Plotting graphs...')
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(len(acc))

# Accuracy vs Epochs
plt.plot(epochs, acc, 'r', label='Training Accuacy')
plt.plot(epochs, val_acc, 'b', label='Validation Accuacy')
plt.title('Accuracy vs Epochs')
plt.legend()
plt.savefig(f'{PLOTS}/accuracy_vs_epochs.jpg')

# Accuracy vs Epochs
plt.figure()
plt.plot(epochs, loss, 'r', label='Training Loss')
plt.plot(epochs, val_loss, 'b', label='Validation Loss')
plt.title('Loss vs Epochs')
plt.legend()
plt.savefig(f'{PLOTS}/loss_vs_epochs.jpg')

# Note ending time
end = time.time()

# Calculate time difference
exec_time = int(end - start)

# Print execution time
print(f'Total execution time: {exec_time}s (' + \
      f'{exec_time // 3600}h {(exec_time // 60) % 60}m {exec_time % 60}s)')