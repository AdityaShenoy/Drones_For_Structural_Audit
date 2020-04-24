from PIL import Image
from re import match
import pyautogui as pag

# Folder paths
FOLDER_PREFIX = 'F:/github/Drones_For_Structural_Audit/dataset'
INPUT_FOLDER = f'{FOLDER_PREFIX}/raw/all_images'
OUTPUT_FOLDER = f'{FOLDER_PREFIX}/internal/256'

# Image count
INPUT_IMG_CNT = 228

# Start the program from this image
IMG_START = 5

# Image dimension
IMG_SIZE = 256

# Grid for red border
GRID = Image.open('F:/github/Drones_For_Structural_Audit/images/grid.png').convert('RGBA')

# Iterate for all images starting from IMG_START
for img_num in range(IMG_START, INPUT_IMG_CNT):

  # Open the image
  img = Image.open(f'{INPUT_FOLDER}/{img_num:03}.jpg')

  # Extract width and height of the image
  w, h = img.width, img.height

  # Visited set to avoid overlap of 2 samples
  vis = set()

  # Height loop
  y = 0
  while y < h - IMG_SIZE + 1:

    # Width loop
    x = 0
    while x < (w - IMG_SIZE + 1):

      # Check if the IMG_SIZE window is already visited
      for xx in range(IMG_SIZE):
        if (x + xx, y) in vis:
          break
      
      # If not visited
      else:

        # Print progress
        print(f'Processing {(x, y)}')

        # Crop the image in the window
        cropped_img = img.crop((x, y, x + IMG_SIZE, y + IMG_SIZE))

        # Scale the image to original image height
        scaled_img = cropped_img.resize((h, h))

        # Create a new image for showing original image and above scaled image
        img_to_show = Image.new(mode='RGB', size=(w+h, h))
        img_to_show.paste(img, box=(0, 0))
        img_to_show.paste(scaled_img, box=(w, 0))

        # Highlight the window with red border grid
        img_to_show.paste(GRID, box=(x, y), mask=GRID)

        # Show user the image
        img_to_show.show()

        # Ask for input
        label = input(f'Enter label (cdnp / 1-{w - IMG_SIZE - x} / x / xx / y1-y{h - IMG_SIZE - y}): ')

        # Change the application tab and close the image
        pag.hotkey('alt', 'tab')
        pag.hotkey('alt', 'f4')

        # If image is labelled
        if label in 'cdnp':

          # Convert cdnp to 0123
          l = 'cdnp'.find(label)

          # Save the sample in the output folder with proper name
          cropped_img.save(f'{OUTPUT_FOLDER}/{img_num:03}_{x:04}_{y:04}_{l}.jpg')

          # Mark the window visited
          for yy in range(IMG_SIZE):
            for xx in range(IMG_SIZE):
              vis.add((x + xx, y + yy))
        
        # Discard the sample and add it to visited set
        elif label == 'x':
          for yy in range(IMG_SIZE):
            for xx in range(IMG_SIZE):
              vis.add((x + xx, y + yy))
        
        # Discard the whole row and add it to visited set
        elif label == 'xx':
          for yy in range(IMG_SIZE):
            for xx in range(w - x + 1):
              vis.add((x + xx, y + yy))
        
        # If input is a number the increment the x value by that value
        elif match('\\d+', label):
          x += int(label) - 1
        
        # If input is y and a number the increment the y value by that value
        elif match('y\\d+', label):
          y += int(label[1:]) - 1
          break
          
        # If invalid input, do not do anything
        else:
          x -= 1
      x += 1
    y += 1