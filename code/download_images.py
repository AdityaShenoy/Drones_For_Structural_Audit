import pyautogui as pag
import time

time.sleep(1) # Time to switch to browser
IMAGE_COORDS = 650, 700
NEXT_COORDS = 1850, 850
n = 0
while True:
  pag.click(IMAGE_COORDS, button=pag.RIGHT)
  time.sleep(0.1) # Time to load the right click context menu
  pag.press('s')
  while pag.position() == IMAGE_COORDS:
    pass # Time to download image and windows file explorer prompt window to load
  pag.typewrite(f'{n}.jpg')
  pag.hotkey('ctrl', 'l')
  pag.typewrite('c:/users/admin/desktop/images\n')
  time.sleep(1) # Time to load the contents of the folder
  pag.hotkey('alt', 's')
  time.sleep(1) # Time to save the image
  pag.click(NEXT_COORDS)
  time.sleep(1) # Time to load next image
  n += 1