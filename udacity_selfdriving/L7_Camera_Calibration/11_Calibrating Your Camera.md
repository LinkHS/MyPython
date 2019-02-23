# Calibrating Your Camera

watch the video


## Note Regarding Corner Coordinates

Since the origin corner is (0,0,0) the final corner is (6,4,0) relative to this corner rather than (7,5,0).

**Examples of Useful Code**

Converting an image, imported by cv2 or the glob API, to grayscale:

 ```python
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
 ```

Note: If you are reading in an image using `mpimg.imread()` this will read in an **RGB** image and you should convert to grayscale using `cv2.COLOR_RGB2GRAY`, but if you are using `cv2.imread()` or the glob API, as happens in this video example, this will read in a **BGR** image and you should convert to grayscale using `cv2.COLOR_BGR2GRAY`. We'll learn more about color conversions later on in this lesson, but please keep this in mind as you write your own code and look at code examples.

Finding chessboard corners (for an 8x6 board):
 ```python
ret, corners = cv2.findChessboardCorners(gray, (8,6), None)
 ```
 
Camera calibration, given object points, image points, and the **shape of the grayscale image**:
 ```python
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
 ```

Undistorting a test image:
 ```python
dst = cv2.undistort(img, mtx, dist, None, mtx)
 ```


### A note on image shape

The shape of the image, which is passed into the **calibrateCamera** function, is just the height and width of the image. One way to retrieve these values is by retrieving them from the **grayscale image shape** array `gray.shape[::-1]`. This returns the image width and height in pixel values like (1280, 960).

Another way to retrieve the image shape, is to get them directly from the color image by retrieving the first two values in the color image shape array using `img.shape[1::-1]`. This code snippet asks for just the first two values in the shape array, and reverses them. Note that in our case we are working with a greyscale image, so we only have 2 dimensions (color images have three, height, width, and depth), so this is not necessary.

It's important to use an entire grayscale image shape or the first two values of a color image shape. This is because the entire shape of a color image will include a third value -- the number of color channels -- in addition to the height and width of the image. For example the shape array of a color image might be (960, 1280, 3), which are the pixel height and width of an image (960, 1280) and a third value (3) that represents the three color channels in the color image which you'll learn more about later, and if you try to pass these three values into the calibrateCamera function, you'll get an error.

```{.python .input  n=62}
b[x:y:z] 相当于 b[start:end:step]，x默认0，y默认-1，z默认1
关于溢出，如果第一次递进超过end，就算溢出


# b = np.zero((1,3,2,4)).shape
b = (1, 3, 2, 4) # 等价于 np.zero((1,3,2,4)).shape
print(b)
print('0:', b[0:0:1])   # 从0开始，到0之前结束，递进1
print('1:', b[0:0:2])   # 从0开始，到0之前结束，递进2
print('2:', b[0:1:2])   # 从0开始，到1之前结束，递进2（溢出）
>>>
0: ()
1: ()
2: (1,)

# ---
print('3:', b[0:-1:2])  # 从0开始，到-1（最后一个）之前结束，递进2
print('4:', b[0::-1])   # 从0开始，递进-1（溢出）
print('5:', b[1::1])    # 从1开始，递进1
print('6:', b[1::-1])   # 从1开始，递进-1
>>>
3: (1, 2)
4: (1,)
5: (3, 2, 4)
6: (3, 1)

# ---
print('7:', b[2::1])    # 从2开始，递进1
print('8:', b[2::-1])   # 从2开始，递进-1
print('9:', b[2::2])    # 从2开始，递进2（溢出）
print('10:', b[1::2])   # 从1开始，递进2
>>>
7: (2, 4)
8: (2, 3, 1)
9: (2,)
10: (3, 4)
    
# ---
print('11:', b[0:])     # 等于b[0::]；从0开始
print('12:', b[0:2])    # 从0开始，到2之前结束
print('13:', b[0:1])    # 从0开始，到1之前结束（溢出，因为默认递进1吧）
print('14:', b[0:3])    # 从0开始，到3之前结束
>>>
11: (1, 3, 2, 4) (1, 3, 2, 4)
12: (1, 3)
13: (1,)
14: (1, 3, 2)

# ---
print('15:', b[0:3:-1]) # 从0开始，到3之前结束，递进-1
print('16:', b[0:3:3])  # 从0开始，到3之前结束，递进3（溢出）
print('17:', b[0:5])    # 从0开始，到5之前结束
print('18:', b[0:3:2])  # 从0开始，到3之前结束，递进2
>>>
15: ()
16: (1,)
17: (1, 3, 2, 4)
18: (1, 2)
```

```{.json .output n=62}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "(1, 3, 2, 4)\n0: ()\n1: ()\n2: (1,)\n3: (1, 2)\n4: (1,)\n5: (3, 2, 4)\n6: (3, 1)\n7: (2, 4)\n8: (2, 3, 1)\n9: (2,)\n10: (3, 4)\n11: (2, 4)\n12: (2,)\n13: (1,)\n14: (1, 3, 2)\n15: ()\n16: (1,)\n17: (1, 3, 2, 4)\n18: (1, 2)\n19: (1,)\n"
 }
]
```

```{.python .input  n=82}
b = (1, 3, 2, 4)
print(b)
print('11:', b[0:], b[0::])    # 从0开始

```

```{.json .output n=82}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "(1, 3, 2, 4)\n11: (1, 3, 2, 4) (1, 3, 2, 4)\n"
 }
]
```
