# Following code checks for overlapping of sub images
# ==============================================================================
with open('F:\github\Drones_For_Structural_Audit\code\input_for_generate_dataset.txt') as f:

  lines = [list(map(int, line.strip().split())) for line in f.readlines()]
  
  for i, (img1, x1, y1, label1) in enumerate(lines[:-1]):
    for img2, x2, y2, label2 in lines[i+1:]:

      if (img1 == img2) and (abs(x1-x2) < 256) and (abs(y1-y2) < 256):
        print(img1, x1, y1, label1, img2, x2, y2, label2)

# ==============================================================================