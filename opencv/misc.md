
## 水平/垂直方向拼接图片
预处理，保证拼接的图片具有相同的通道数。因为拼接方向上维度可以不一致，但是其他维度必须一致。

例如，1280x720 和 1200x720 只能在水平方向上拼接（因为他们的高一样，宽不一样）。  
同理1280x720 和 1280x600 只能在垂直方向上拼接。
```
grey = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
# Make the grey scale image have three channels
grey_3_channel = cv2.cvtColor(grey, cv2.COLOR_GRAY2BGR)
```

```
# 垂直方向拼接
numpy_vertical = np.vstack((image, grey_3_channel))
numpy_vertical_concat = np.concatenate((image, grey_3_channel), axis=0)

# 水平方向拼接
numpy_horizontal = np.hstack((image, grey_3_channel))
numpy_horizontal_concat = np.concatenate((image, grey_3_channel), axis=1)
```

---
## 图片存储可能不是连续的
```
image = np.ascontiguousarray(image, dtype=np.uint8)
```

---
## 读取图片
```
# 注意此时img.dtype为uint8，channel为RGB
img = matplotlib.image.imread('pic.jpg')
```

---
## 灰色度图转彩色
img_corlor = np.dstack((img_mono, img_mono, img_mono))
img_corlor = cv2.cvtColor(img_mono, cv2.COLOR_GRAY2RGB)

---
## 放大缩小；Resize
```
# Resized the image to a quarter of its original size
image = cv2.resize(image, (0, 0), None, .25, .25)

# Resize to 800x600 (width*height*)
image = cv2.resize(image, (800, 600), None)
```

