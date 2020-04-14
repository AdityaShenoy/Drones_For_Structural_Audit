from os import scandir, system
from PIL import Image
from threading import Thread
from time import sleep

# This is the function that the thread will process
# Each thread will process one class as specified in the parameter
def process_images(class_):

  # Count for all subimages of the current class
  all_img_cnt = 0

  # Scan the class folder
  with scandir(f'{INPUT_FOLDER}\\{class_}') as folder:

    # Set the thread message
    thread_message[class_] = f'Class {class_}\n' + \
      'Calculating the number of images...\n\n'

    # For all files in the folder
    for file in folder:

      # Open the image file
      img = Image.open(file.path)

      # Extract dimensions of the image
      w, h = img.width, img.height

      # Count all the subimages that will be generated for this class
      all_img_cnt += (h - IMAGE_SIZE + 1) * (w - IMAGE_SIZE + 1)
    
  # Number of digits to name the files with same length numbers (0 padded)
  num_digits = len(str(all_img_cnt))

  # Counter for naming images
  cntr = 0

  # Scan the class folder
  with scandir(f'{INPUT_FOLDER}\\{class_}') as folder:

    # For all files in the folder
    for file in folder:

      # Open the image file
      img = Image.open(file.path)

      # Extract dimensions of the image
      w, h = img.width, img.height

      # The count of subimages of the current image
      cur_img_cnt = (h - IMAGE_SIZE + 1) * (w - IMAGE_SIZE + 1)

      # Counter for current sub image count
      cur_cntr = 0

      # Iterate through all possible windows of size IMAGE_SIZE * IMAGE_SIZE
      for i in range(h - IMAGE_SIZE + 1):
        for j in range(w - IMAGE_SIZE + 1):

          # Set thread message
          thread_message[class_] = f'Class: {class_}\n' + \
            f'Cur img progress: {cur_cntr}/{cur_img_cnt} ({cur_cntr/cur_img_cnt*100:.2f}%)\n' + \
            f'All img progress: {cntr}/{all_img_cnt} ({cntr/all_img_cnt*100:.2f}%)\n'

          # Crop the image in the window
          cropped_img = img.crop((j, i, j + IMAGE_SIZE, i + IMAGE_SIZE))

          # Save the cropped_img in the class subfolder of the OUPUT_FOLDER
          cropped_img.save(f'{OUPUT_FOLDER}\\{class_}\\{cntr:0{num_digits}}.jpg')

          # Incremenet the counters
          cntr += 1
          cur_cntr += 1
  
  # Set thread message
  thread_message[class_] = f'Class: {class_}\n' + \
    f'Processing of {all_img_cnt} images done!\n\n'

  # Sleep for 1 second
  sleep(1)

  # Signal the main thread that the processing is completed
  thread_is_completed[class_] = True

# c = Crack
# d = Dampness
# p = Paint peel off
# n = No defect
CLASS_FOLDERS = ['c', 'd', 'p', 'n']

# Input folders and output folders
INPUT_FOLDER = 'C:\\Users\\admin\\Desktop\\input_aditya'
OUPUT_FOLDER = 'C:\\Users\\admin\\Desktop\\output_aditya'

# Output image dimension
IMAGE_SIZE = 256

# Flags to track whether the thread is running or is completed
thread_is_completed = {c: False for c in CLASS_FOLDERS}

# Thread progress messages
thread_message = {c: '' for c in CLASS_FOLDERS}

# For all class folders
for class_ in CLASS_FOLDERS:

  # Start the thread for each class 
  Thread(target=process_images, args=(class_, )).start()

# While all the threads have not completed their processing
while not all([v for k, v in thread_is_completed.items()]):

  # Sleep for 1 second
  sleep(1)

  # Clear the screen
  system('cls')

  # For all class threads
  for class_ in CLASS_FOLDERS:

    # Print the thread messages
    print(thread_message[class_])