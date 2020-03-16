from os import scandir
from PIL import Image

# c = Crack
# d = Dampness
# p = Paint peel off
# n = No defect
# x = Not a wall
CLASS_FOLDERS = ['c', 'd', 'p', 'n', 'x']

# Input folders and output folders
INPUT_FOLDER = 'C:\\Users\\admin\\Desktop\\label_input\\'
OUPUT_FOLDER = 'C:\\Users\\admin\\Desktop\\labelled\\'

# Output image dimension
IMAGE_SIZE = 256

# For all class folders
for class_ in CLASS_FOLDERS:

  # Scan the class folder
  with scandir(f'{INPUT_FOLDER}{class_}') as folder:

    # Count for subimages of the current class
    cnt = 0

    # For all files in the folder
    for file in folder:

      # Open the image file
      img = Image.open(file.path)

      # Extract dimensions of the image
      w, h = img.width, img.height

      # Iterate through all possible windows of size IMAGE_SIZE * IMAGE_SIZE
      for i in range(h - IMAGE_SIZE + 1):
        for j in range(w - IMAGE_SIZE + 1):

          # Crop the image in the window
          cropped_img = img.crop((j, i, j + IMAGE_SIZE, i + IMAGE_SIZE))

          # Save the cropped_img in the class subfolder of the OUPUT_FOLDER
          cropped_img.save(f'{OUPUT_FOLDER}{class_}\\{cnt}.jpg')

          # Incremenet the count
          cnt += 1