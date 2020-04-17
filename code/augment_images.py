from os import scandir
from PIL import Image

# Input and output folder paths
INPUT_FOLDER = 'F:\\github\\Drones_For_Structural_Audit\\dataset\\internal\\300'
OUTPUT_FOLDER = 'F:\\github\\Drones_For_Structural_Audit\\dataset\\internal\\300_aug'

# Sub folder names
FOLDERS = ['all', 'train', 'test']

# For all class folders
for class_ in 'cdnp':

  # For all sub folders
  for FOLDER in FOLDERS:

    # Count of images
    img_cnt = 0

    # Scan the folder and its files
    with scandir(f'{INPUT_FOLDER}\\{FOLDER}\\{class_}') as folder:
      for file in folder:

        # Each image would be mapped to 8 images
        img_cnt += 8

    # Number of digits in image count
    num_digits = len(str(img_cnt))

    # Counter for images
    img_cntr = 0

    # Scan the folder and its files
    with scandir(f'{INPUT_FOLDER}\\{FOLDER}\\{class_}') as folder:
      for file in folder:
    
        # Open the image
        img = Image.open(file.path)

        # Folder prefix for this class
        FOLDER_PREFIX = f'{OUTPUT_FOLDER}\\{FOLDER}\\{class_}'

        # Save the original image
        img.save(f'{FOLDER_PREFIX}\\{img_cntr:0{num_digits}}.jpg')

        # Increment the image counter
        img_cntr += 1

        # For all augmentation operations
        for op in [Image.ROTATE_90, Image.ROTATE_180, Image.ROTATE_270,
                    Image.FLIP_TOP_BOTTOM, Image.FLIP_LEFT_RIGHT,
                    Image.TRANSPOSE, Image.TRANSVERSE]:

          # Save the original image
          img.transpose(op).save(f'{FOLDER_PREFIX}\\{img_cntr:0{num_digits}}.jpg')

          # Increment the image counter
          img_cntr += 1