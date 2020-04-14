from os import scandir
from shutil import copyfile

PATH_PREFIX = 'F:\\github\\Drones_For_Structural_Audit\\dataset\\camera\\'
OUTPUT_FOLDER = 'F:\\github\\Drones_For_Structural_Audit\\dataset\\camera\\all_images\\'
subfolders = ['aditya', 'rahul', 'kalyani']

i = 0

for subfolder in subfolders:
  with scandir(f'{PATH_PREFIX}{subfolder}') as folder:
    for file in folder:
      copyfile(file.path, f'{OUTPUT_FOLDER}{i:>03}.jpg')
      i += 1