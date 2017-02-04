import numpy as np
import cv2
from matplotlib import pyplot as plt

img = cv2.imread('../data/messi5.jpg', 0)
plt.imshow(img, cmap='gray', interpolation='bicubic')
plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
plt.show()

cv2.imshow('image', img)
k = cv2.waitKey(0)
if k == 27:  # wait for ESC key to exit
    cv2.destroyAllWindows()
elif k == ord('s'):  # wait for 's' key to save and exit
    cv2.imwrite('messigray.png', img)
    cv2.destroyAllWindows()
