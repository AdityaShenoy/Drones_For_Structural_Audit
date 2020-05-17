from PIL import Image
import numpy as np
import tensorflow as tf
import time
import os
import shutil
import pickle
import cv2

# Constants
IS_LIVE = False
IMG_SIZE = 256
NUM_CHANNELS = 3
COLOR_CODE = [Image.new('RGB', (IMG_SIZE, IMG_SIZE), x)
                for x in [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]] #cdnp
CLASSES = 'cdnp'
FPS = 1
VIDEO_DIM = (512, 512) if IS_LIVE else (1280, 1024)

# File and folder paths
DESKTOP = 'C:/Users/admin/Desktop/content'
CONTENT = '/content' if not os.path.exists(DESKTOP) else DESKTOP
DRIVE = f'{CONTENT}/drive/My Drive'
DRIVE_MODEL_ZIP = f'{DRIVE}/model.zip'
MODEL = f'{CONTENT}/model'
MODEL_ZIP = f'{MODEL}.zip'
LAYERS = f'{MODEL}/layers.txt'
WEIGHTS = f'{MODEL}/weights'
DRIVE_ALL_IMAGES_ZIP = f'{DRIVE}/all_images.zip'
ALL_IMAGES = f'{CONTENT}/all_images'
ALL_IMAGES_ZIP = f'{ALL_IMAGES}.zip'
VIDEO_EXTENSION = 'avi'
INPUT_VIDEO = f'{CONTENT}/test.mp4'
OUTPUT = f'{CONTENT}/output'
OUTPUT_VIDEO = f'{OUTPUT}/test.{VIDEO_EXTENSION}'
OUTPUT_ZIP = f'{OUTPUT}.zip'
DRIVE_OUTPUT_ZIP = f'{DRIVE}/output.zip'

# For all following folders,
# if the folders already exists delete the folder and make the folders and subfolders
print('Deleting old folders and making new empty folders...')
for FOLDER in [MODEL, OUTPUT]:
  if os.path.exists(FOLDER):
    shutil.rmtree(FOLDER, onerror=lambda a,b,c:0)
    time.sleep(1)
  os.mkdir(FOLDER)
for src, inter, dst in [(DRIVE_MODEL_ZIP, MODEL_ZIP, MODEL)]:
  shutil.copy(src, CONTENT)
  shutil.unpack_archive(inter, dst)
  os.unlink(inter)

# This function performs sliding window mechanism
# and predicts the class of each window
def predict(img):
  # COnverts pixels to image object
  img = Image.fromarray(img)
  # Round down the dimensions of the image
  w, h = img.width // IMG_SIZE * IMG_SIZE, img.height // IMG_SIZE * IMG_SIZE
  img = img.resize((w, h))
  # Make a new result image
  result_img = Image.new(mode='RGB', size=(2 * w, h) if w < h else (w, 2 * h))
  result_img.paste(img, box=(0, 0))
  # Sliding window mechanism
  for y in range(0, h, IMG_SIZE):
    for x in range(0, w, IMG_SIZE):
      window = img.crop((x, y, x + IMG_SIZE, y + IMG_SIZE))
      model_output = model.predict(
        np.asarray(window).reshape(1, IMG_SIZE, IMG_SIZE, 3))
      prediction = model_output[0].argmax()
      window_result = COLOR_CODE[prediction]
      result_img.paste(window_result, box=(w + x, y) if w < h else (x, h + y))
  return np.asarray(result_img)

# Load the model
print('Loading the model...')
with open(LAYERS) as f:
  model = eval(f.read())
with open(WEIGHTS, 'rb') as f:
  weights = pickle.load(f)
  model.set_weights(weights)

# # For all following folders,
# # if the folders already exists delete the folder and make the folders and subfolders
# print('Deleting old camera images and unzipping the images from drive...')
# if os.path.exists(ALL_IMAGES):
#   shutil.rmtree(ALL_IMAGES, onerror=lambda a,b,c:0)
# os.mkdir(ALL_IMAGES)
# shutil.copy(DRIVE_ALL_IMAGES_ZIP, CONTENT)
# shutil.unpack_archive(ALL_IMAGES_ZIP, ALL_IMAGES)
# os.unlink(ALL_IMAGES_ZIP)
# # For all camera images, predict on the sliding window mechanism
# print('Predicting on camera images...')
# with os.scandir(ALL_IMAGES) as folder:
#   for file_num, file in enumerate(folder, start=1):
#     img = Image.open(file.path)
#     result_img = predict(img)
#     result_img.save(f'{OUTPUT}/{file.name}')
#     print(f'Processed {file_num} images...')

# Testing the sliding window mechanism on video
print(f'Testing the sliding window mechanism on {"live" if IS_LIVE else "test"} video...')
out = cv2.VideoWriter(OUTPUT_VIDEO, 0, FPS, VIDEO_DIM)
avg_time_per_frame, num_frames = None, 0
cap = cv2.VideoCapture(0 if IS_LIVE else INPUT_VIDEO)
while cap.isOpened():
  start = time.time()
  ret, frame = cap.read()
  if not ret:
    break
  result = cv2.cvtColor(predict(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)), cv2.COLOR_RGB2BGR)
  if IS_LIVE:
    cv2.imshow('frame', result)
    if cv2.waitKey(1000 // 60) & 0xFF == ord('q'):
      break
  end = time.time()
  out.write(result)
  if avg_time_per_frame:
    avg_time_per_frame = (avg_time_per_frame * num_frames + (end - start)) / \
                          (num_frames + 1)
    num_frames += 1
  else:
    avg_time_per_frame, frames = end-start, 1
  print(f'{avg_time_per_frame} seconds')
cap.release()
out.release()
cv2.destroyAllWindows()

# Archive and store output in drive
print('Archiving output folder and storing it in drive...')
shutil.make_archive(OUTPUT, 'zip', OUTPUT)
if os.path.exists(DRIVE_OUTPUT_ZIP):
  os.unlink(DRIVE_OUTPUT_ZIP)
shutil.move(OUTPUT_ZIP, DRIVE)