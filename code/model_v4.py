import os
import shutil
import tensorflow as tf
import numpy as np
from PIL import Image
import time
import matplotlib.pyplot as plt

# Note starting time
start = time.time()

# Constants
IMG_SIZE = 256
KERAS_DISTORTION_SCALE = 2
DATASET_SIZE = 4 * 625 * 8 * KERAS_DISTORTION_SCALE
TRAIN_PROP = 0.8
TRAINING_SIZE = int(DATASET_SIZE * TRAIN_PROP)
VALIDATING_SIZE = DATASET_SIZE - TRAINING_SIZE
BATCH_SIZE = 32
CLASSES = 'cdnp'

# Folder and file paths
CONTENT = '/content'
DRIVE = f'{CONTENT}/drive/My Drive'
DATASET = f'{CONTENT}/dataset'
MODEL = f'{CONTENT}/model'
TRAIN = f'{DATASET}/train'
VALIDATE = f'{DATASET}/validate'
PLOTS = f'{MODEL}/plots'
SAVE = f'{MODEL}/save/'
DRIVE_DATASET_ZIP = f'{DRIVE}/dataset.zip'
DATASET_ZIP = f'{DATASET}.zip'
ARCHITECTURE = f'{PLOTS}/model.jpg'
ACCURACY = f'{PLOTS}/accuracy_vs_epochs.jpg'
LOSS = f'{PLOTS}/loss_vs_epochs.jpg'
DRIVE_MODEL_ZIP = f'{DRIVE}/model.zip'
MODEL_ZIP = f'{MODEL}.zip'

# For all following folders,
# if the folders already exists delete the folder and make the folders and subfolders
print('Deleting old folders and making new empty folders...')
for FOLDER in [DATASET, MODEL, PLOTS]:
  if os.path.exists(FOLDER):
    shutil.rmtree(FOLDER, onerror=lambda a,b,c:0)
  os.mkdir(FOLDER)
shutil.copy(DRIVE_DATASET_ZIP, CONTENT)
shutil.unpack_archive(DATASET_ZIP, DATASET)
os.unlink(DATASET_ZIP)

# Initialize the ML model
print('Building model...')
model = tf.keras.models.Sequential([
  tf.keras.layers.Conv2D(filters=16, kernel_size=(3,3), activation='relu',
                         input_shape=(IMG_SIZE, IMG_SIZE, 3)),
  tf.keras.layers.MaxPooling2D(2, 2),
  tf.keras.layers.Conv2D(filters=16, kernel_size=(3,3), activation='relu'),
  tf.keras.layers.MaxPooling2D(2, 2),
  tf.keras.layers.Conv2D(filters=16, kernel_size=(3,3), activation='relu'),
  tf.keras.layers.MaxPooling2D(2, 2),
  tf.keras.layers.Conv2D(filters=16, kernel_size=(3,3), activation='relu'),
  tf.keras.layers.MaxPooling2D(2, 2),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(32, activation='relu'),
  tf.keras.layers.Dense(len(CLASSES), activation='softmax')
])

# Visualize the model
tf.keras.utils.plot_model(model, to_file=ARCHITECTURE, show_shapes=True)

# Complile the model
print('Compiling the mdoel...')
model.compile(
  loss = 'categorical_crossentropy',
  optimizer = tf.keras.optimizers.Adam(lr=0.001),
  metrics = ['accuracy']
)

# Set up the training and validating dataset generator
print('Setting up dataset generators...')
train_gen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1/255)\
              .flow_from_directory(directory = TRAIN)
validate_gen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1/255)\
              .flow_from_directory(directory = VALIDATE)

# Training the model
print('Training the model...')
history = model.fit(
  x = train_gen,
  epochs = 15,
  verbose = 1,
  validation_data = validate_gen,
  steps_per_epoch = TRAINING_SIZE // BATCH_SIZE,
  validation_steps = VALIDATING_SIZE // BATCH_SIZE
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
plt.savefig(ACCURACY)

# Accuracy vs Epochs
plt.figure()
plt.plot(epochs, loss, 'r', label='Training Loss')
plt.plot(epochs, val_loss, 'b', label='Validation Loss')
plt.title('Loss vs Epochs')
plt.legend()
plt.savefig(LOSS)

# Save model
model.save(SAVE)

# Archive the model and copy to drive
print('Archiving the model folder and storing it in drive...')
shutil.make_archive(MODEL, 'zip', MODEL)
if os.path.exists(DRIVE_MODEL_ZIP):
  os.unlink(DRIVE_MODEL_ZIP)
shutil.move(MODEL_ZIP, DRIVE)

# Note ending time
end = time.time()

# Calculate time difference
exec_time = int(end - start)

# Print execution time
print(f'Total execution time: {exec_time}s (' + \
      f'{exec_time // 3600}h {(exec_time // 60) % 60}m {exec_time % 60}s)')