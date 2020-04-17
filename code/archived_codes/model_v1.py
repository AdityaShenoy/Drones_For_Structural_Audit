import tensorflow as tf
import numpy as np
from PIL import Image
from os import scandir
from time import time

# Note start time
start = time()

# Constants
CLASSES = 'cdnp'
IMAGE_SIZE = 256

# Folder paths
FOLDER_PREFIX = 'F:\\github\\Drones_For_Structural_Audit\\dataset\\internal'

INPUT_FOLDER = f'{FOLDER_PREFIX}\\300'
INPUT_ALL_FOLDER = f'{INPUT_FOLDER}\\all'
INPUT_TRAIN_FOLDER = f'{INPUT_FOLDER}\\train'
INPUT_TEST_FOLDER = f'{INPUT_FOLDER}\\test'

OUTPUT_FOLDER = f'{FOLDER_PREFIX}\\{IMAGE_SIZE}'
OUTPUT_ALL_FOLDER = f'{OUTPUT_FOLDER}\\all'
OUTPUT_TRAIN_FOLDER = f'{OUTPUT_FOLDER}\\train'
OUTPUT_TEST_FOLDER = f'{OUTPUT_FOLDER}\\test'

IMAGE_FOLDER = 'F:\\github\\Drones_For_Structural_Audit\\images'

# Function for printing the progress
def progress(msg):
  print(f'{"="*100}\n\n{msg}\n\n{"="*100}')

# Print progress
progress(f'Preparing model...')

# Initialize a sequential model
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3)),
  tf.keras.layers.Dense(32),
  tf.keras.layers.Dense(4)
])

# Compile the model
model.compile(
  loss = 'sparse_categorical_crossentropy',
  optimizer = 'sgd',
  metrics = ['accuracy']
)

# Visualize the model
tf.keras.utils.plot_model(model, f'{IMAGE_FOLDER}\\model.jpg', show_shapes=True)

# Print progress
progress(f'Training the model...')

# For all classes
for label, class_ in enumerate(CLASSES):

  # Print progress
  progress(f'Calculating total {class_} images...')

  # Count of images of this class
  total_imgs = 0

  # For all files in the training folder
  with scandir(f'{INPUT_TRAIN_FOLDER}\\{class_}') as folder:
    for file in folder:

      # Open the image file
      img = Image.open(file.path)

      # Extract dimensions of the image
      w, h = img.width, img.height

      # Count the images
      total_imgs += (w - IMAGE_SIZE + 1) * (h - IMAGE_SIZE + 1)
  
  # Print progress
  progress(f'Training the model with {class_} images...')

  # Count number of digits in the total images for output file names
  num_digits = len(str(total_imgs))

  # Counter for image
  img_cntr = 0

  # For all files in the training folder
  with scandir(f'{INPUT_TRAIN_FOLDER}\\{class_}') as folder:
    for file in folder:

      # Open the image file
      img = Image.open(file.path)

      # Extract dimensions of the image
      w, h = img.width, img.height

      # Calculate sub images obtained from this image
      total_sub_imgs = (w - IMAGE_SIZE + 1) * (h - IMAGE_SIZE + 1)

      # For all possible windows
      for y in range(h - IMAGE_SIZE + 1):

        # Initialize empty X data
        X = []

        # Construct label data
        Y = np.asarray([label for _ in range(w - IMAGE_SIZE + 1)])

        for x in range(w - IMAGE_SIZE + 1):

          # Crop the image in the window
          cropped_img = img.crop((x, y, x + IMAGE_SIZE, y + IMAGE_SIZE))

          # Append data to X
          X.append(np.asarray(cropped_img))

          # Save the cropped image in the output training folder
          # file_name = f'{OUTPUT_TRAIN_FOLDER}\\{class_}\\{img_cntr:0{num_digits}}.jpg'
          # cropped_img.save(file_name)

          # Increment the image counter
          img_cntr += 1
      
        # Convert list to array and normalize the data
        X = np.asarray(X) / 255

        # Train the model with X and Y
        model.train_on_batch(X, Y)

        # Debugging weight
        progress(f'{model.get_weights()}')

# Print progress
progress(f'Training of model finished...\nTesting the model...')

# For all classes
for label, class_ in enumerate(CLASSES):

  # Print progress
  progress(f'Calculating total {class_} images...')

  # Count of images of this class
  total_imgs = 0

  # For all files in the test folder
  with scandir(f'{INPUT_TEST_FOLDER}\\{class_}') as folder:
    for file in folder:

      # Open the image file
      img = Image.open(file.path)

      # Extract dimensions of the image
      w, h = img.width, img.height

      # Count the images
      total_imgs += (w - IMAGE_SIZE + 1) * (h - IMAGE_SIZE + 1)
  
  # Print progress
  progress(f'Testing the model with {class_} images...')

  # Count number of digits in the total images for output file names
  num_digits = len(str(total_imgs))

  # Counter for image
  img_cntr = 0

  # For all files in the testing folder
  with scandir(f'{INPUT_TEST_FOLDER}\\{class_}') as folder:
    for file in folder:

      # Open the image file
      img = Image.open(file.path)

      # Extract dimensions of the image
      w, h = img.width, img.height

      # Calculate sub images obtained from this image
      total_sub_imgs = (w - IMAGE_SIZE + 1) * (h - IMAGE_SIZE + 1)

      # Initialize empty X data
      X = []

      # Construct label data
      Y = np.asarray([label for _ in range(total_sub_imgs)])

      # For all possible windows
      for y in range(h - IMAGE_SIZE + 1):
        for x in range(w - IMAGE_SIZE + 1):

          # Crop the image in the window
          cropped_img = img.crop((x, y, x + IMAGE_SIZE, y + IMAGE_SIZE))

          # Append data to X
          X.append(np.asarray(cropped_img))

          # Save the cropped image in the output test folder
          # file_name = f'{OUTPUT_TEST_FOLDER}\\{class_}\\{img_cntr:0{num_digits}}.jpg'
          # cropped_img.save(file_name)

          # Increment the image counter
          img_cntr += 1
      
      # Convert list to array and normalize the data
      X = np.asarray(X) / 255

      # Test the model with X and Y
      losses_and_metrics = model.evaluate(X, Y)

      # Print progress
      progress(f'Loss and metrics of current batch: {losses_and_metrics}')

# Note end time
end = time()

# Calculate execution time
exec_time = int(end - start)

# Print time
print(f'{exec_time}s')
print(f'{exec_time // 3600}h {(exec_time // 60) % 60}m {exec_time % 60}s')