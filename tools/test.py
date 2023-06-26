import cv2
from PIL import Image
from skimage import io
import matplotlib.pyplot as plt
import numpy as np

img_path = "../CVC/training/images/30.tif"
mask_path = "../CVC/training/masks/30.tif"
original_img = cv2.imread(img_path)
original_img = cv2.resize(original_img, (240, 240), interpolation=cv2.INTER_AREA)
original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
original_mask = cv2.imread(mask_path)
original_mask = cv2.resize(original_mask, (240, 240), interpolation=cv2.INTER_AREA)
original_mask = cv2.cvtColor(original_mask, cv2.COLOR_BGR2GRAY)
cv2.imwrite("../output_image/30.png", original_img)
cv2.imwrite("../output_image/mask_30.png", original_mask)



