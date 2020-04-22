from PIL import Image
from re import match
import pyautogui as pag
FOLDER_PREFIX = 'F:/github/Drones_For_Structural_Audit/dataset'
INPUT_FOLDER = f'{FOLDER_PREFIX}/raw/all_images'
OUTPUT_FOLDER = f'{FOLDER_PREFIX}/internal/256'
INPUT_IMG_CNT = 228
IMG_START = 3
IMG_SIZE = 256
GRID = Image.open('F:/github/Drones_For_Structural_Audit/images/grid.png').convert('RGBA')
for img_num in range(IMG_START, INPUT_IMG_CNT):
  img = Image.open(f'{INPUT_FOLDER}/{img_num:03}.jpg')
  w, h = img.width, img.height
  vis = set()
  y = 0
  while y < h - IMG_SIZE + 1:
    x = 0
    while x < (w - IMG_SIZE + 1):
      for xx in range(IMG_SIZE):
        if (x + xx, y) in vis:
          break
      else:
        print(f'Processing {(x, y)}')
        cropped_img = img.crop((x, y, x + IMG_SIZE, y + IMG_SIZE))
        scaled_img = cropped_img.resize((h, h))
        img_to_show = Image.new(mode='RGB', size=(w+h, h))
        img_to_show.paste(img, box=(0, 0))
        img_to_show.paste(scaled_img, box=(w, 0))
        img_to_show.paste(GRID, box=(x, y), mask=GRID)
        img_to_show.show()
        label = input(f'Enter label (cdnp / 1-{w - IMG_SIZE - x} / x / xx / y1-y{h - IMG_SIZE - y}): ')
        pag.hotkey('alt', 'tab')
        pag.hotkey('alt', 'f4')
        if label in 'cdnp':
          l = 'cdnp'.find(label)
          cropped_img.save(f'{OUTPUT_FOLDER}/{img_num:03}_{x:04}_{y:04}_{l}.jpg')
          for yy in range(IMG_SIZE):
            for xx in range(IMG_SIZE):
              vis.add((x + xx, y + yy))
        elif label == 'x':
          for yy in range(IMG_SIZE):
            for xx in range(IMG_SIZE):
              vis.add((x + xx, y + yy))
        elif label == 'xx':
          for yy in range(IMG_SIZE):
            for xx in range(w - x + 1):
              vis.add((x + xx, y + yy))
        elif match('\\d+', label):
          x += int(label) - 1
        elif match('y\\d+', label):
          y += int(label[1:]) - 1
          break
        else:
          x -= 1
      x += 1
    y += 1