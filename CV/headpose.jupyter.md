# 头部姿态估计

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt
```

## 人脸关键点3D到2D投影

![](http://static.zybuluo.com/AustinMxnet/of7yqo16opt6t5ldyagsqh9e/image.png)

如图所示，我们知道人脸的3D信息，例如平均脸的6个关键点坐标如下：

```python
face3D_points = np.array([
    (0.0, 0.0, 0.0),             # Nose tip
    (0.0, -330.0, -65.0),        # Chin
    (-225.0, 170.0, -135.0),     # Left eye left corner
    (225.0, 170.0, -135.0),      # Right eye right corne
    (-150.0, -150.0, -125.0),    # Left Mouth corner
    (150.0, -150.0, -125.0)      # Right mouth corner
])
```

3D人脸投影到2D图像上的对应关键点位置我们可以通过“人脸关键点”算法求得，于是问题就变成了已知n个3D空间点以及它们的2D投影位置时，如何估计相机（或人脸）所在的位姿。下面读入示例图片和对应的关键点位置：

```python
img = cv2.imread("data/face2.png")
face2D_points = np.loadtxt("data/face2.txt", delimiter=',')

for p in face2D_points:
    cv2.circle(img, (int(p[0]), int(p[1])), 3, (0, 0, 255), -1)

plt.figure(figsize=(12, 14))    
plt.imshow(img[:,:,::-1])
plt.show()
```

人脸关键点通过相机成像从3D到2D的转换公式如下：

$$
\left(\begin{array}{l}
u \\
v \\
1
\end{array}\right)=\frac{1}{Z}\left(\begin{array}{ccc}
f_{x} & 0 & c_{x} \\
0 & f_{y} & c_{y} \\
0 & 0 & 1
\end{array}\right)
\left(\begin{array}{llll}
r_{11} & r_{12} & r_{13} & t_{1} \\
r_{21} & r_{22} & r_{23} & t_{2} \\
r_{31} & r_{32} & r_{33} & t_{3}
\end{array}\right)
\left(\begin{array}{l}
X \\
Y \\
Z \\
1
\end{array}\right)
$$

相机的内参可以通过标定求得（或者估算），剩下的$R,t$就是我们需要的姿态信息。

## 相机内参估计

通常来说可以通过标定来求得相机的内参。但是如果不是自己的相机拍摄的照片（例如网上的），我们很难得知其内参。由于我们不关心尺度信息（只关心姿态角），并且$f_x, f_y$通常相差也不大，所以可以用一个较大的值代替焦距$f_x, f_y$。**个人猜测：注意这样做的前提是图片没有被裁剪过，即$c_x, c_y$是准确的，也就是人脸在图像中的位置就是原始照片中的位置。**

```python
size = img.shape
fx = fy = size[1]
cx, cy = size[1]/2, size[0]/2
# Assuming no lens distortion
dist_coeffs = np.zeros((4, 1))

camera_matrix = np.array(
    [[fx, 0,  cx],
     [0,  fy, cy],
     [0,  0,  1 ]], dtype="double"
)

camera_matrix
```

## Perspective-n-Point

PnP（Perspective-n-Point）是求解3D到2D点对运动的方法。PnP问题有很多种求解方法，例如用三对点估计位姿的P3P、直接线性变换（DLT）、EPnP（Efficient PnP）、UPnP等等）。此外，还能用非线性优化的方式，构建最小二乘问题并迭代求解，也就是万金油式的Bundle Adjustment。本文用OpenCV的`solvePnP()`：

```python
(success, rotation_vector, translation_vector) = cv2.solvePnP(
    face3D_points, face2D_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)

print("Rotation Vector:\n {0}".format(rotation_vector))
print("\nTranslation Vector:\n {0}".format(translation_vector))
```

`solvePnP()`输出的是旋转向量（Rotation Vector）和位移向量（Translation Vector），我们可以将一个3D坐标轴通过这两个向量投影到图像上，观察其姿态变化：

```python
# Project a 3D point (0, 0, 1000.0) onto the image plane.
# We use this to draw a line sticking out of the nose
xyz_3d = np.array([(500.0, 0.0, 0.0), (0.0, 500.0, 0.0), (0.0, 0.0, 500.0)])
(xyz_2d, jacobian) = cv2.projectPoints(xyz_3d, rotation_vector,
                                       translation_vector, camera_matrix, dist_coeffs)

colors = ((0, 200, 0), (200, 50, 0), (0, 200, 200))
o = tuple(face2D_points[0].astype(int))
for uv, color in zip(xyz_2d.astype(int), colors):
    cv2.line(img, o, tuple(uv[0]), color, 2)

plt.figure(figsize=(12, 14))
plt.imshow(img[:, :, ::-1])
plt.show()
```

黄色为人脸朝向，蓝色为人脸中轴线，绿色为水平线，从图中看出效果还是不错的（毕竟相机参数是估计的）。我们可以将平均脸的关键点3D坐标投影到图中看下是否准确：

```python
(meanface_2d, jacobian) = cv2.projectPoints(face3D_points, rotation_vector,
                                            translation_vector, camera_matrix, dist_coeffs)

for uv in meanface_2d.astype(int):
    cv2.circle(img, tuple(uv[0]), 3, (0, 255, 255), -1)

plt.figure(figsize=(12, 14))
plt.imshow(img[:, :, ::-1])
plt.show()
```

## 欧拉角
旋转向量（Rotation Vector）不是很直观，我们可以转成欧拉角来表示。欧拉角包括三个方向上的旋转，分别为Pitch、Roll、Yaw：

![](http://static.zybuluo.com/AustinMxnet/ofbfvdy5749e55yqx85q5tsg/image.png)

由于我们求得的是旋转向量`rotation_vector`，需要先转为旋转矩阵`rvec_matrix`，然后通过分解投影矩阵`proj_matrix`得到欧拉角：

```python
import math

rvec_matrix = cv2.Rodrigues(rotation_vector)[0]
proj_matrix = np.hstack((rvec_matrix, translation_vector))
eulerAngles = cv2.decomposeProjectionMatrix(proj_matrix)[6]

pitch, yaw, roll = [math.radians(_) for _ in eulerAngles]

pitch = math.degrees(math.asin(math.sin(pitch)))
roll = math.degrees(math.asin(math.sin(roll)))
yaw = math.degrees(math.asin(math.sin(yaw)))

pitch, roll, yaw
```

从结果可以看出，驾驶员头部抬起ptich=21.89度（方向似乎与上图中的pitch方向相反），向左歪头roll=9.49度，向左偏转yaw=30.85度
