
## 读取图片
```
# 注意此时img.dtype为uint8，channel为RGB
img = matplotlib.image.imread('pic.jpg')
```

---
## 灰色度图转彩色
img_corlor = np.dstack((img_mono, img_mono, img_mono))
img_corlor = cv2.cvtColor(img_mono, cv2.COLOR_GRAY2RGB)
