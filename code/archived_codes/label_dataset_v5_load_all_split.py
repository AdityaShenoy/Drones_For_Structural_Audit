from os import scandir, listdir, system
import numpy as np
from PIL import Image
import pickle

# This is the folder which contains image for testing and training
IMAGE_FOLDER = 'F:\\github\\Drones_For_Structural_Audit\\dataset\\internal\\output'

# Training proportion
TRAIN_PROP = 0.8

# Initialize empty training and testing inputs and outputs (X and Y)
x_train, y_train, x_test, y_test = [], [], [], []

# Progress message
prog_msg = 'Preparing image data...'
system('cls')
print(prog_msg)

# For 4 classes
for label, class_ in enumerate('cdnp'):

  # Calculate the total number of images in the folder
  num_img = len(listdir(f'{IMAGE_FOLDER}\\{class_}'))

  # Number of digits in file name
  num_dig = len(str(num_img))
  
  # This is the split point of training and testing images
  train_test_split = int(TRAIN_PROP * num_img)

  # Progress message
  prog_msg += f'\nPreparing training images of class {class_}...'
  system('cls')
  print(prog_msg)

  # For all training images
  for i in range(train_test_split):

    # Progress message
    system('cls')
    print(f'{prog_msg}\n{i}/{train_test_split}...')

    # Add image data to training input
    x_train.append(
      np.asarray(
        Image.open(f'{IMAGE_FOLDER}\\{class_}\\{i:0{num_dig}}.jpg')
      )
    )

    # Add label to training output
    y_train.append(label)

  # Progress message
  prog_msg += f'\nTraining images of class {class_} prepared...' + \
              f'\nPreparing testing images of class {class_}...'
  system('cls')
  print(prog_msg)

  # For all testing images
  for i in range(train_test_split, num_img):

    # Progress message
    system('cls')
    print(f'{prog_msg}\n{i-train_test_split}/{num_img - train_test_split}...')

    # Add image data to testing input
    x_test.append(
      np.asarray(
        Image.open(f'{IMAGE_FOLDER}\\{class_}\\{i:0{num_dig}}.jpg')
      )
    )

    # Add label to testing output
    y_test.append(label)

  # Progress message
  prog_msg += f'\nTesting images of class {class_} prepared...'
  system('cls')
  print(prog_msg)

# Progress message
prog_msg += f'\nImage data prepared...'
system('cls')
print(prog_msg)

# Convert the lists to np arrays
x_train, y_train, x_test, y_test = np.asarray(x_train), np.asarray(y_train), \
                                   np.asarray(x_test), np.asarray(y_test)

# Pickle all the objects for avoiding repititive computations
with open('x_train', 'wb') as f:
  pickle.dump(x_train, f)
with open('x_test', 'wb') as f:
  pickle.dump(x_test, f)
with open('y_train', 'wb') as f:
  pickle.dump(y_train, f)
with open('y_test', 'wb') as f:
  pickle.dump(y_test, f)

print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)