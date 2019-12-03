# -*- coding: utf-8 -*-
# +
from pylab import *
import numpy

import camera
import homography
# -

# # 载入数据
# 首先，我们使用齐次坐标来表示这些点，points里存储的三维齐次坐标（第四维为1）

# +
points = homography.make_homog(loadtxt('house.p3d').T)

points.shape, points[:,0], points[:,1]
# -

# 然后我们使用一个投影矩阵来创建 Camera 对象将这些三维点投影到图像平面（二维齐次坐标，第三维为1）并执行绘制操作。
#
# $x = norm(P\times points)$

# +
P = hstack((eye(3), array([[0], [0], [-10]]))) #投影矩阵
cam = camera.Camera(P)
x = cam.project(points)

print(P, '\n\n', x.shape)
print(['%.2f'%x for x in points[:,0]])
print(['%.2f'%x for x in x[:,0]])
# -

figure()
plot(x[0], x[1], 'k.')


# # 相机移动

# +
from scipy.linalg import expm, norm

def M(axis, theta):
    return expm(numpy.cross(numpy.eye(3), axis/norm(axis)*theta))

axis, theta = [0, 0, 1], 0.1
M0 = M(axis, theta)

print(numpy.dot(M0,v))
# [ 2.74911638  4.77180932  1.91629719]

r = 0.05 * numpy.random.rand(3)
r = numpy.array([1, 0, 0])
rot = camera.rotation_matrix(r)

M0 = numpy.append(M0, [[0,0,0]], axis=0)
M0 = numpy.append(M0, numpy.array([[0,0,0,1]]).T, axis=1)

M0, rot
# -

figure()
for t in range(1):
#   cam.P = dot(cam.P, rot)
  cam.P = dot(cam.P, M0)
  x = cam.project(points)
  plot(x[0], x[1], 'k.')

# +
import scipy
# scipy.linalg.expm?
# -



show()


