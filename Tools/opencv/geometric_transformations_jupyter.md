## Goals
Learn to apply different geometric transformation to images like translation, rotation, affine transformation etc.

You will see these functions: `cv.getPerspectiveTransform`

---
## Transformations
OpenCV provides two transformation functions, `cv.warpAffine` and `cv.warpPerspective`, with which you can have all kinds of transformations. `cv.warpAffine` takes a 2x3 transformation matrix while `cv.warpPerspective` takes a 3x3 transformation matrix as input.

---
### Scaling
Scaling is just resizing of the image. OpenCV comes with a function `cv.resize()` for this purpose. The size of the image can be specified manually, or you can specify the scaling factor. Different interpolation methods are used. Preferable interpolation methods are `cv.INTER_AREA` for shrinking and `cv.INTER_CUBIC` (slow) & `cv.INTER_LINEAR` for zooming. By default, interpolation method used is cv.INTER_LINEAR for all resizing purposes. You can resize an input image either of following methods:

```{.python .input  n=1}
import cv2

img = cv2.imread('data/messi5.jpg', 0)
print('1:', img.shape)

res = cv2.resize(img, None, fx=2, fy=2, interpolation = cv2.INTER_CUBIC)
print('1:', res.shape)

#OR

height, width = img.shape[:2]
res = cv2.resize(img, (2*width, 2*height), interpolation = cv2.INTER_CUBIC)
print('2:', res.shape)
```

---
### Translation
Translation is the shifting of object's location. If you know the shift in (x,y) direction, let it be $(t_x,t_y)$, you can create the transformation matrix $M$ as follows:

$$M = \begin{bmatrix} 1 & 0 & t_x \\ 0 & 1 & t_y \end{bmatrix}$$

You can take make it into a Numpy array of type **np.float32** and pass it into `cv.warpAffine()` function. See below example for a shift of (100,50):
> Third argument of the `cv.warpAffine()` function is the size of the output image, which should be in the form of **(width, height)**. Remember width = number of columns, and height = number of rows.

```{.python .input}
%reset -f
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
%matplotlib inline

img = cv.imread('data/messi5.jpg', 0)
rows, cols = img.shape

M = np.float32([[1, 0, 100], 
                [0, 1, 50]])

dst = cv.warpAffine(img, M, (cols, rows))
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 9))
f.tight_layout()
ax1.imshow(img)
ax2.imshow(dst)
plt.show()
```

---
### Rotation
Rotation of an image for an angle $\theta$ is achieved by the transformation matrix of the form
$$
M = \begin{bmatrix} cos\theta & -sin\theta \\ −sin\theta & cos\theta \end{bmatrix}
$$

But OpenCV provides scaled rotation with adjustable center of rotation so that you can rotate at any location you prefer. Modified transformation matrix is given by

$$\begin{bmatrix} \alpha & \beta & (1- \alpha ) \cdot center.x - \beta \cdot center.y \\ - \beta & \alpha & \beta \cdot center.x + (1- \alpha ) \cdot center.y \end{bmatrix}$$

where:  

$$\begin{array}{l} \alpha = scale \cdot \cos \theta , \\ \beta = scale \cdot \sin \theta \end{array}$$

To find this transformation matrix, OpenCV provides a function, `cv.getRotationMatrix2D`. Check below example which rotates the image by 90 degree with respect to center without any scaling.

```{.python .input  n=10}
%reset -f
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
%matplotlib inline

img = cv.imread('data/messi5.jpg', 0)
rows, cols = img.shape
M = cv.getRotationMatrix2D((cols / 2, rows / 2), 90, 1)
dst = cv.warpAffine(img, M, (cols, rows))

f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 9))
f.tight_layout()
ax1.imshow(img, cmap='gray')
ax2.imshow(dst, cmap='gray')
plt.show()
```

---
### Affine Transformation
In affine transformation, all parallel lines in the original image will still be parallel in the output image. To find the transformation matrix, we need three points from input image and their corresponding locations in output image. Then `cv.getAffineTransform` will create a 2x3 matrix which is to be passed to `cv.warpAffine`.

Check below example, and also look at the points I selected (which are marked in Green color):

```{.python .input}
%reset -f
import numpy as np
import cv2
import matplotlib.pyplot as plt
%matplotlib inline

img = np.zeros((512, 512, 3), np.uint8)
rows, cols, _ = img.shape

# Draw Lines
for row, col in zip(np.linspace(0, rows, 10), np.linspace(0, cols, 10)):
    row = int(row); col = int(col)
    cv2.line(img, (0, row), (int(cols), row), (255, 0, 0), 2)
    cv2.line(img, (col, 0), (col, int(rows)), (255, 0, 0), 2)


pts1 = np.float32([[50,  50], [200, 50], [ 50, 200]]) # 原坐标
pts2 = np.float32([[10, 100], [200, 50], [100, 250]]) # 转换后坐标
M = cv2.getAffineTransform(pts1, pts2)
dst = cv2.warpAffine(img, M, (cols, rows))


# Draw pts1, pts2 on original and affined image respectively
for p1, p2 in zip(pts1, pts2):
    cv2.circle(img, tuple(p1), 4, (0, 0, 255), -1)
    cv2.circle(dst, tuple(p2), 4, (0, 255, 0), -1)
    
    
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 9))
f.tight_layout()
ax1.imshow(img)
ax2.imshow(dst)
plt.show()
```

---
### Perspective Transformation
For perspective transformation, you need a 3x3 transformation matrix. Straight lines will remain straight even after the transformation. To find this transformation matrix, you need 4 points on the input image and corresponding points on the output image. Among these 4 points, 3 of them should not be collinear. Then transformation matrix can be found by the function `cv.getPerspectiveTransform`. Then apply `cv.warpPerspective` with this 3x3 transformation matrix.

See the code below:

```{.python .input  n=1}
%reset -f
import numpy as np
import cv2
import matplotlib.pyplot as plt
%matplotlib inline

img = np.zeros((512, 512, 3), np.uint8)
rows, cols, _ = img.shape

pts1 = np.float32([[100, 400], [500, 500], [200, 100], [300, 100]]) # 源坐标，4个蓝色点
pts2 = np.float32([[100, 400], [500, 500], [100, 100], [400, 100]]) # 转换后坐标，4个绿色小圈

# Draw Lines
cv2.line(img, tuple(pts1[0]), tuple(pts1[2]), (255, 0, 0), 3, cv2.LINE_AA) # 左红色直线
cv2.line(img, tuple(pts1[1]), tuple(pts1[3]), (255, 0, 0), 3, cv2.LINE_AA) # 右红色直线

cv2.line(img, tuple(pts1[3]), tuple(pts1[0]), (255, 255, 0), 2, cv2.LINE_AA) # 绿色直线
cv2.ellipse(img, (100, 400), (50, 300), 30, 180, 260, (255, 255, 0), 1, cv2.LINE_AA) # 绿色曲线

M = cv2.getPerspectiveTransform(pts1, pts2)
dst = cv2.warpPerspective(img, M, (cols, rows))

# Draw pts1, pts2 on original and affined image respectively
for p1, p2 in zip(pts1, pts2):
    cv2.circle(img, tuple(p1), 4, (0, 0, 255), -1, cv2.LINE_AA)
    cv2.circle(img, tuple(p2), 4, (0, 255, 0), 1, cv2.LINE_AA)

f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 9))
f.tight_layout()
ax1.imshow(img)
ax2.imshow(dst)
plt.show()
```
