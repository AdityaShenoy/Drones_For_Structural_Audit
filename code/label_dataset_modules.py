from PIL import Image
import numpy as np

def resize(img, width, height):
  return img.resize((width, height))

def paste(original, new, left, top, mask=None):
  original.paste(new, (left, top), mask)

def gen_outline(width, height):

  OUTLINE_WIDTH = 5
  red = np.array([255, 0, 0, 255], dtype=np.uint8)
  t = np.tile(red, OUTLINE_WIDTH * width)
  l = np.tile(red, 5)
  m = np.full((width - 2 * OUTLINE_WIDTH) * 4, 0, dtype=np.uint8)
  lmr = np.tile(np.hstack((l, m, l)), height-2*OUTLINE_WIDTH)
  tlmrd = np.hstack((t, lmr, t))
  final = np.reshape(tlmrd, (height, width, 4))
  return Image.fromarray(final)


def crop(img, left, top, width, height):
  return img.crop((left, top, left + width, top + height))


def save(img, dest):
  img.save(dest)


def copy(img):
  return img.copy()


def process(img, i, j, h, w, height, width, img_to_show_temp, file_name, WINDOW_SIZE):

  # Create a window outline image
  window_outline = gen_outline(w, h)

  # Crop the image in the window
  cropped_img = crop(img, j, i, w, h)

  # Scale the cropped image to square of size original img height
  scaled_img = resize(cropped_img, height, height)

  # Copy the temp image to show
  img_to_show = copy(img_to_show_temp)

  # Paste the scaled image to the right side
  paste(img_to_show, scaled_img, width, 0)

  # Paste the window outline as a mask at window location
  paste(img_to_show, window_outline, j, i, window_outline)

  # Save the image to the show folder
  save(img_to_show, f'../dataset/temp/show/{file_name}')

  # Crop the image in the window to WINDOW_SIZE sized sqaure
  final_img = resize(cropped_img, WINDOW_SIZE, WINDOW_SIZE)

  # Save the final image to the crop folder
  save(final_img, f'../dataset/temp/crop/{file_name}')