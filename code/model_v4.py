import os
import shutil
import tensorflow as tf
import numpy as np
from PIL import Image
import time
import matplotlib.pyplot as plt
import pickle
from sklearn.metrics import classification_report

# Note starting time
start = time.time()

# Constants
IMG_SIZE = 256
KERAS_DISTORTION_SCALE = 2
DATASET_SIZE = 4 * 625 * 8
TRAIN_SPLIT = 0.8
TRAINING_SIZE = round(DATASET_SIZE * TRAIN_SPLIT * KERAS_DISTORTION_SCALE)
VALIDATING_SIZE = round(DATASET_SIZE * (1 - TRAIN_SPLIT))
BATCH_SIZE = 32
CLASSES = 'cdnp'
NUM_CHANNELS = 3

# Folder and file paths
DESKTOP = 'C:/Users/admin/Desktop/content'
CONTENT = '/content' if not os.path.exists(DESKTOP) else DESKTOP
DRIVE = f'{CONTENT}/drive/My Drive'
DRIVE_DATASET_ZIP = f'{DRIVE}/dataset.zip'
DATASET = f'{CONTENT}/dataset'
DATASET_ZIP = f'{DATASET}.zip'
TRAIN = f'{DATASET}/train'
VALIDATE = f'{DATASET}/validate'
MODEL = f'{CONTENT}/model'
WEIGHTS = f'{MODEL}/weights'
METRICS = f'{MODEL}/metrics.txt'
PLOTS = f'{MODEL}/plots'
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

# Copying the zip file from drive to colab file system
print('Copying and extracting the zip file from drive to colab file system...')
shutil.copy(DRIVE_DATASET_ZIP, CONTENT)
shutil.unpack_archive(DATASET_ZIP, DATASET)
os.unlink(DATASET_ZIP)

# Initialize the ML model
print('Building model...')
model = tf.keras.models.Sequential([
  tf.keras.layers.Conv2D(filters=16, kernel_size=(3,3), activation='relu',
                         input_shape=(IMG_SIZE, IMG_SIZE, NUM_CHANNELS)),
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

# Compile the model
print('Compiling the model...')
model.compile(
  loss = 'categorical_crossentropy',
  optimizer = tf.keras.optimizers.Adam(lr=0.001),
  metrics = ['accuracy']
)

# Set up the training, validating and testing dataset generator
print('Setting up dataset generators...')
train_gen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1/255)\
              .flow_from_directory(directory = TRAIN)
validate_gen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1/255)\
              .flow_from_directory(directory = VALIDATE)
test_gen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1/255)\
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

# Testing the model
print('Testing the model...')
labels, predictions = [], []
for i, (x, y) in enumerate(test_gen):
  if i == VALIDATING_SIZE // BATCH_SIZE:
    break
  labels.extend(y.argmax(axis=1))
  pred = model.predict(
    x = x,
    verbose = 0,
    steps = 1
  )
  predictions.extend(pred.argmax(axis=1))
labels, predictions = np.asarray(labels), np.asarray(predictions)
with open(METRICS, 'w') as f:
  f.write(f'{tf.math.confusion_matrix(labels, predictions)}\n')
  f.write(f'{classification_report(labels, predictions)}')

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

# Loss vs Epochs
plt.figure()
plt.plot(epochs, loss, 'r', label='Training Loss')
plt.plot(epochs, val_loss, 'b', label='Validation Loss')
plt.title('Loss vs Epochs')
plt.legend()
plt.savefig(LOSS)

# Save model weights
w = model.get_weights()
with open(WEIGHTS, 'wb') as f:
  pickle.dump(w, f)

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