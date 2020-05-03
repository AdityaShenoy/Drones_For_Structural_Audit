from PIL import Image
import threading
import os
import shutil
import time

# Folder paths
ALL_IMAGES_FOLDER = 'F:/github/Drones_For_Structural_Audit/dataset/raw/all_images'
INPUT_FOLDER = 'F:/github/Drones_For_Structural_Audit/dataset/internal/samples/input'
OUTPUT_FOLDER = 'F:/github/Drones_For_Structural_Audit/dataset/internal/samples/output'

# Image dimension
IMG_SIZE = 256

# Thread function
def f(img, i):

  # Open the image
  img = Image.open(img)

  # For all assigned windows
  for y in range(i*img.height//4, (i+1)*img.height//4):
    for x in range(img.width - IMG_SIZE + 1):

      # Crop the image in the window
      cropped_img = img.crop((x, y, x + IMG_SIZE, y + IMG_SIZE))

      # Save the cropped image in the input folder
      cropped_img.save(f'{INPUT_FOLDER}/{img_file.name[:4]}_{y:04}_{x:04}_.jpg')

      # Print progress
      thread_msg[i] = f'Processed {(y - (i*img.height//4)) * (w - IMG_SIZE + 1) + x + 1}' + \
            f' / {(img.height//4) * (w - IMG_SIZE + 1)}'

  # Finish the thread
  thread_msg[i] = 'Thread work completed'
  time.sleep(1)
  thread_completed[i] = True


# Scan the all images folder and iterate for all files in it
with os.scandir(ALL_IMAGES_FOLDER) as img_folder:
  for img_file in img_folder:

    # Note starting time
    start = time.time()

    # Open the image file
    img = Image.open(img_file.path)

    # Extract dimensions of the image
    w, h = img.width, img.height

    # Thread progress message and completion flags
    thread_msg = {t_id: '' for t_id in range(4)}
    thread_completed = {t_id: False for t_id in range(4)}

    # Divide the work among 4 threads
    for i in range(4):
      threading.Thread(target=f, args=(img_file.path, i)).start()
    
    # While all threads have not finished their works
    while not all(thread_completed.values()):
      os.system('cls')
      print(*thread_msg.items(), sep='\n')
      time.sleep(1)
    
    # Note ending time
    end = time.time()

    # Calculate time difference
    exec_time = int(end - start)

    # Print execution time
    print(f'Total execution time: {exec_time}s (' + \
          f'{exec_time // 3600}h {(exec_time // 60) % 60}m {exec_time % 60}s)')

    # Loop infinitely
    while True:

      # Trim the sample folder images or proceed for next image
      inp = input('trim / next: ')

      # If input is trim
      if inp == 'trim':

        # Scan the output folder and iterate for all files in the folder
        with os.scandir(OUTPUT_FOLDER) as out_folder:
          for out_file in out_folder:

            # Extract image number and (x, y) coordinates from the 
            img_num, y, x = map(int, out_file.name.split('_')[:3])

            # Images trimmed in this iteration
            trim_cnt = 0

            # For all overlapping images
            for yy in range(max(0, y - IMG_SIZE + 1), min(y + IMG_SIZE - 1, h - IMG_SIZE) + 1):
              for xx in range(max(0, x - IMG_SIZE + 1), min(x + IMG_SIZE - 1, w - IMG_SIZE) + 1):

                # If the path exists
                if os.path.exists(f'{INPUT_FOLDER}/{img_num}_{yy:04}_{xx:04}_.jpg'):

                  # Delete the file
                  os.unlink(f'{INPUT_FOLDER}/{img_num}_{yy:04}_{xx:04}_.jpg')

                  # Increment the trim count
                  trim_cnt += 1
        
        # Print the number of images trimmed
        print(f'Trimmed {trim_cnt} images')
      
      # If input is next
      elif inp == 'next':

        # Remove all the previous images from the folder
        shutil.rmtree(INPUT_FOLDER)
        os.mkdir(INPUT_FOLDER)

        # Break the loop to proceed to next image
        break