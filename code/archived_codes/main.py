from PIL import Image

# Function for getting kernel score
def apply_border_kernel(image):

  # Load array of image
  a = image.load()

  # Initialize kernel_score
  kernel_score = 0

  # For row in array
  for p in range(ML_INPUT_SIZE):

    # for column in array
    for q in range(ML_INPUT_SIZE):
      
      # To check if it is on border
      if p in [0, ML_INPUT_SIZE - 1] or q in [0, ML_INPUT_SIZE - 1]:

        # sum of RGB channel
        kernel_score += sum(a[q,p])

  # Return kernel score
  return kernel_score


# Declare constants
MIN_WINDOW_SIZE = 1 # This can match to the dimension of the data set images
MAX_WINDOW_SIZE = 100 # This can be minimum of image width and height
WINDOW_INCREASE_STEP = 1
KERNEL_THRESHOLD = 0
ML_INPUT_SIZE = 256


# Initialize window size to minimum
window_size = MIN_WINDOW_SIZE

# Top left coordinate of the moving window
i, j = 0, 0

# Image of the infrastructure
image = Image.open('')

# Initializing empty result list
result = []

# Loop until window size reaches MAX_WINDOW_SIZE
while (window_size <= MAX_WINDOW_SIZE):

  # Crop the image to the window from i, j as top left coordinates
  # and height and width as window size
  cropped_image = image.crop((j, i, j + window_size, i + window_size))

  # Scale the cropped image to dimensions suitable for machine learning input
  scaled_image = cropped_image.resize((ML_INPUT_SIZE, ML_INPUT_SIZE))

  # Apply border kernel to the scaled image and get a score
  kernel_score = apply_border_kernel(scaled_image)

  # If the kernel score exceeds threshold
  if (kernel_score >= KERNEL_THRESHOLD):

    # Pass the image to ML model
    ml_result = detect_crack(scaled_image)

    # If the result is positive meaning crack is detected
    if ml_result:

      # Measure the width of the crack
      crack_width = measure_crack_width(scaled_image)
      
      # Store the crack width and position of window
      result.append((i, j, window_size, crack_width))

      # Mask the window portion of the original image to
      # avoid duplicate detection of cracks by bigger window
      masked_image = mask_image(image, i, j, window_size)
    
  # If the window has reached to extreme right of the image
  if j == image.width:

    # If the window has reached to bottom right end of the image
    if i == image.height:

      # Increase the size of the window by step value
      window_size += WINDOW_INCREASE_STEP

      # Move the window to top left corner
      i, j = 0, 0
    
    # If there are more rows to pass through
    else:

      # Move to next row and reset column to left
      i, j = i + 1, 0
  
  # If the window has not reached the rightmost end
  else:

    # Move the window to right
    j += 1



