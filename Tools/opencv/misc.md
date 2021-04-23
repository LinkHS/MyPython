## 翻转图片

```python
image_flipped = np.fliplr(image)
# or
image_flipped = cv2.flip(image, 1)
```

## 水平/垂直方向拼接图片

预处理，保证拼接的图片具有相同的通道数。因为拼接方向上维度可以不一致，但是其他维度必须一致。

例如，1280x720 和 1200x720 只能在水平方向上拼接（因为他们的高一样，宽不一样）。同理1280x720 和 1280x600 只能在垂直方向上拼接。

```python
grey = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
# Make the grey scale image have three channels
grey_3_channel = cv2.cvtColor(grey, cv2.COLOR_GRAY2BGR)
```

```python
# 垂直方向拼接
numpy_vertical = np.vstack((image, grey_3_channel))
numpy_vertical_concat = np.concatenate((image, grey_3_channel), axis=0)

# 水平方向拼接
numpy_horizontal = np.hstack((image, grey_3_channel))
numpy_horizontal_concat = np.concatenate((image, grey_3_channel), axis=1)
```

---
## 图片存储可能不是连续的
```python
image = np.ascontiguousarray(image, dtype=np.uint8)
```

---
## 读取图片
```python
# 注意此时img.dtype为uint8，channel为RGB
img = matplotlib.image.imread('pic.jpg')
# or
img = ndimage.imread(current_path) # from scipy import ndimage
```

### Load a paletted PNG
```
from PIL import Image

im_pil = Image.open("mandril_color.png")
print(im_pil.mode) # Plattle 
>>>
P


#---
im_cv = np.asarray(im_pil)
cv2.open("mandril_color.png").shape # it's been converted to 3-channel color
>>>
(512, 512, 3)

im_cv.shape # it's become grayscale
(512, 512)
```

---
## 灰色度图转彩色
```python
img_corlor = np.dstack((img_mono, img_mono, img_mono))
# or
img_corlor = cv2.cvtColor(img_mono, cv2.COLOR_GRAY2RGB)
```

---
## 放大缩小；Resize
```python
# Resized the image to a quarter of its original size
image = cv2.resize(image, (0, 0), None, .25, .25)

# Resize to 800x600 (width*height*)
image = cv2.resize(image, (800, 600), None)
```

