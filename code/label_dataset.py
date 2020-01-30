from PIL import Image
from os import system, scandir, listdir, name
from threading import Thread
from time import sleep
import label_dataset_modules as ldm

# Function which is run by threads
def f(tid):

  # Iterate through all image URLS allotted to this thread
  for img_url in alloted_work[tid]:

    # Open the image in the URL
    img = Image.open(img_url)

    # Round down the new dimensions of the image such
    # that they are a multiple of WINDOW_SIZE each
    width = (img.width // WINDOW_SIZE) * WINDOW_SIZE
    height = (img.height // WINDOW_SIZE) * WINDOW_SIZE

    # Calculating possible subimages count of the current image
    m, n = width // WINDOW_SIZE, height // WINDOW_SIZE
    subimage_cnt[tid] = m * n * (m + 1) * (n + 1) // 4
    
    # Resize the image to above dimensions
    img = ldm.resize(img, width, height)

    # Create a new image to be shown
    img_to_show_temp = Image.new('RGB', (width + height, height))

    # Paste the original image to the left side
    ldm.paste(img_to_show_temp, img, 0, 0)

    # Iterate for rows
    for i in range(0, height - WINDOW_SIZE + 1, WINDOW_SIZE):

      # Iterate for cols
      for j in range(0, width - WINDOW_SIZE + 1, WINDOW_SIZE):

        # For all heights
        for h in range(WINDOW_SIZE, height - i + 1, WINDOW_SIZE):

          # For all widths
          for w in range(WINDOW_SIZE, width - j + 1, WINDOW_SIZE):

            # File name for the image to show and final image
            file_name = f'{tid + NO_OF_THREADS * completed_work[tid][0]:03}_{completed_work[tid][1]:05}.jpg'

            # Process the sub image
            ldm.process(img, i, j, h, w, height, width, img_to_show_temp, file_name, WINDOW_SIZE)

            # Increment the sub image count
            completed_work[tid][1] += 1

    # Increment the image count
    completed_work[tid][0] += 1

    # Reset the subimage count
    completed_work[tid][1] = 0


# The smallest unit size of window
WINDOW_SIZE = 256

# Number of threads
NO_OF_THREADS = 4

# Initialize empty work allotment list
alloted_work = [[] for _ in range(NO_OF_THREADS)]

# Initialize completed work list to all 0s
completed_work = [[0, 0] for _ in range(NO_OF_THREADS)]

# The sub image count of current image
subimage_cnt = [0 for _ in range(NO_OF_THREADS)]

# If temp show folder is not empty
temp_show_folder = '../dataset/temp/show'
if len(listdir(temp_show_folder)):

  # Empty the temp show folder
  system(f'{"del" if name == "nt" else "rm"} ../dataset/temp/show/*.jpg')

# If temp crop folder is not empty
temp_crop_folder = '../dataset/temp/crop'
if len(listdir(temp_crop_folder)):

  # Empty the temp crop folder
  system(f'{"del" if name == "nt" else "rm"} ../dataset/temp/crop/*.jpg')

# Counter for thread
tid = 0

# Iterate through folder names where camera photos are stored
for dataset_folder in ['aditya', 'rahul', 'kalyani']:

  # Open the folder
  with scandir(f'../dataset/camera/{dataset_folder}') as folder:

    # Iterate through all files in the folder
    for file in folder:

      # Assign the work to the thread
      alloted_work[tid].append(f'../dataset/camera/{dataset_folder}/{file.name}')

      # Increment the thread ID to assign next work
      tid = (tid + 1) % NO_OF_THREADS

# For all thread ids
for i in range(NO_OF_THREADS):

  # Create thread and start it
  Thread(target=f, args=(i,)).start()

# Loop
while True:

  # Check for all threads
  for tid in range(NO_OF_THREADS):

    # If any thread has not completed its alloted work
    if completed_work[tid][0] != len(alloted_work[tid]):

      # Break the for loop and continue the while loop
      break
  
  # If all threads have completed their alloted work
  else:
    
    # Break the while loop and end the program
    break

  # Clear the screen
  system('cls' if name == 'nt' else 'clear')

  # For all threads
  for tid in range(NO_OF_THREADS):

    # Images completed by this thread
    i = completed_work[tid][0]

    # Subimages completed by this thread
    si = completed_work[tid][1]

    # Total images allotted to this thread
    ti = len(alloted_work[tid])

    # Total subimages allotted to this thread
    tsi = subimage_cnt[tid]

    # Print the progress of the thread
    print(f'{tid}:{" " * 10}{i} / {ti} ({i / ti * 100:.2f}%){" " * 10}{si} / {tsi} ({si / tsi * 100:.2f}%)')