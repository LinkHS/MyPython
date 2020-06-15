# skimage
`$ pip install scikit-image`

**注意 skimage 的这些操作会将 image 转化为 float 型（0-1）之间**

```python
import skimage as sk
import numpy as np
import matplotlib.pyplot as plt

image = np.ones((100,120,3))
image[50:100,:,:] = 0.5

plt.imshow(image)
```

## Rotate
Rotate image by a certain angle around its center

```python
from skimage import transform

plt.imshow(sk.transform.rotate(image, 10))
```

## Noise
Add random noise of various types to a floating-point image

```python
from skimage import util

plt.imshow(sk.util.random_noise(image))
```

```python

```

```python

```
