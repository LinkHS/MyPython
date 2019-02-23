# skimage
`$ pip install scikit-image`

**注意 skimage 的这些操作会将 image 转化为 float 型（0-1）之间**
```
import skimage as sk
import numpy

image = np.zeros((100,100,3))
```

## Rotate
Rotate image by a certain angle around its center
```
from skimage import transform

sk.transform.rotate(image, 10)
```

## Noise
Add random noise of various types to a floating-point image
```
from skimage import util

sk.util.random_noise(image)
```

