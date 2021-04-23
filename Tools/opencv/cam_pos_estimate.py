import numpy as np
import cv2
import os
import math

def splitfn(fn):
    path, fn = os.path.split(fn)
    name, ext = os.path.splitext(fn)
    return path, name, ext

filepath = unicode('/Users/austin/Downloads/vlcsnap-2016-12-29-14h19m46s255.png', 'utf8')
img = cv2.imread(filepath, 0)
img1 = cv2.imread(filepath, 0)
pattern_size = (4, 6)
pattern_points = np.zeros((np.prod(pattern_size), 3), np.float32)
pattern_points[:, :2] = np.indices(pattern_size).T.reshape(-1, 2)
found, corners = cv2.findChessboardCorners(img, pattern_size)

if found:
    print 'found'
    imagePoints = corners.reshape(-1, 2)
    objectPoints = pattern_points
    print 'imagePoints', imagePoints
    print 'objectPoints', objectPoints

    camera_matrix = np.array([[1405.02, 0, 646],
                     [0, 1482.6, 401],
                     [0, 0, 1]])

    dist_coefs = np.array([-0.43885165076373917, 0.010025822862121924, -0.00010026880617983947, -0.0010377790918911073])
    retval, rvec, tvec = cv2.solvePnP(objectPoints, imagePoints, camera_matrix, dist_coefs)
    print 'retval: ', retval
    print 'rvec: ', rvec
    print 'tvec: ', tvec

    rot, J = cv2.Rodrigues(-rvec)
    print 'rot: ', rot
    #tvec_gnd = rot.matmul(tvec)
    tvec_gnd = np.matmul(rot, -tvec)
    print 'ground tvec: ', tvec_gnd

    ## transform coordinate
    v = rvec
    sita = math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])
    qw = math.cos(sita / 2)
    ss = math.sin(sita / 2)
    qx = v[0] * ss / sita
    qy = v[1] * ss / sita
    qz = v[2] * ss / sita
    rot_y = math.atan2(2 * qy * qw - 2 * qx * qz, 1 - 2 * qy * qy - 2 * qz * qz)
    rot_z = math.asin(2 * qx * qy + 2 * qz * qw)
    rot_x = math.atan2(2 * qx * qw - 2 * qy * qz, 1 - 2 * qx * qx - 2 * qz * qz)
    print rot_x * 180 / math.pi, rot_y * 180 / math.pi, rot_z * 180 / math.pi


    term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 30, 0.1)
    cv2.drawChessboardCorners(img1, pattern_size, corners, found)
    cv2.imshow('before', img1)
    #cv2.drawMarker()
    # to determine the corners' positions more accurately
    cv2.cornerSubPix(img, corners, (5, 5), (-1, -1), term)
    cv2.drawChessboardCorners(img, pattern_size, corners, found)
    cv2.imshow('after', img)
    cv2.waitKey(0)
else:
    print 'not found'