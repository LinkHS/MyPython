import cv2
import numpy as np
from glob import glob

img_mask = '/Users/austin/work/temp/res/*.png'
img_names = glob(img_mask)

for fn in img_names:
    #print('processing %s... ' % fn)
    img1 = cv2.imread(fn, 0)
    img2 = cv2.imread(fn.replace('res', 'label'), 0)
    same_ft = np.sum(cv2.bitwise_and(img1, img2))
    or_ft = np.sum(cv2.bitwise_or(img1, img2))
    print same_ft, ' ', or_ft, ' ', or_ft/(same_ft - or_ft)