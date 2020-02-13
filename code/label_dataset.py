from PIL import Image
from os import system, scandir, listdir, name
from threading import Thread
from time import sleep

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
    img = img.resize((width, height))

    # Image number
    img_num = f'{tid + NO_OF_THREADS * completed_work[tid][0]:03}'

    # Save the image to img folder with img num as its file name
    img.save(f'{img_folder}\\{img_num}.jpg')

    # Iterate for rows
    for i in range(0, height - WINDOW_SIZE + 1, WINDOW_SIZE):

      # Iterate for cols
      for j in range(0, width - WINDOW_SIZE + 1, WINDOW_SIZE):

        # For all heights
        for h in range(WINDOW_SIZE, height - i + 1, WINDOW_SIZE):

          # For all widths
          for w in range(WINDOW_SIZE, width - j + 1, WINDOW_SIZE):

            subimg_num = f'{completed_work[tid][1]:05}'

            # File name for the final image
            file_name = f'{crop_folder}\\{img_num}_{subimg_num}_{j}_{i}_{w}_{h}.jpg'

            # Process the sub image
            cropped_img = img.crop((j, i, j + w, i + h))

            # Resize the image to WINDOW_SIZE * WINDOW_SIZE
            scaled_img = cropped_img.resize((OUTPUT_SIZE, OUTPUT_SIZE))

            # Save the scaled image
            scaled_img.save(file_name)

            # Increment the sub image count
            completed_work[tid][1] += 1

    # Increment the image count
    completed_work[tid][0] += 1

    # Reset the subimage count
    completed_work[tid][1] = 0


# The smallest unit size of window
WINDOW_SIZE = 512

# Output dimension
OUTPUT_SIZE = 256

# Number of threads
NO_OF_THREADS = 4

# Folder paths
crop_folder = 'F:\\github\\Drones_For_Infrastructure_Crack_Detection\\dataset\\temp\\crop'
img_folder = 'F:\\github\\Drones_For_Infrastructure_Crack_Detection\\dataset\\temp\\img'
camera_folder = 'F:\\github\\Drones_For_Infrastructure_Crack_Detection\\dataset\\camera'

# Initialize empty work allotment list
alloted_work = [[] for _ in range(NO_OF_THREADS)]

# Initialize completed work list to all 0s
completed_work = [[0, 0] for _ in range(NO_OF_THREADS)]

# The sub image count of current image
subimage_cnt = [0 for _ in range(NO_OF_THREADS)]

# Print porgress
print('Deleting old files...')

# If crop folder is not empty
if len(listdir(crop_folder)):

  # Empty the crop folder
  system(f'{"del" if name == "nt" else "rm"} {crop_folder}\\*.jpg')

# If img folder is not empty
if len(listdir(img_folder)):

  # Empty the crop folder
  system(f'{"del" if name == "nt" else "rm"} {img_folder}\\*.jpg')

# Print progress
print('Old files deleted successfully...')

# Counter for thread
tid = 0

# Iterate through folder names where camera photos are stored
for dataset_folder in ['aditya', 'rahul', 'kalyani']:

  # Open the folder
  with scandir(f'{camera_folder}\\{dataset_folder}') as folder:

    # Iterate through all files in the folder
    for file in folder:

      # Assign the work to the thread
      alloted_work[tid].append(f'{camera_folder}\\{dataset_folder}\\{file.name}')

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

  # Sleep for 1 second
  sleep(1)

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
    
    # If the subimage count has not been initialized yet
    if not tsi:
      continue

    # Print the progress of the thread
    print(f'{tid}:{" " * 10}{i} / {ti} ({i / ti * 100:.2f}%){" " * 10}{si} / {tsi} ({si / tsi * 100:.2f}%)')
