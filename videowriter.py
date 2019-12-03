import cv2
import numpy as np
import glob

img_array = []

for filename in sorted(glob.glob('./run1/*.jpg')):
    img = cv2.imread(filename)
    height, width, layers = img.shape
    size = (width,height)
    img_array.append(img)

out = cv2.VideoWriter('run1.mp4',cv2.VideoWriter_fourcc(*'mp4v'), 45, size)
 
for i in range(len(img_array)):
    out.write(img_array[i])
out.release()