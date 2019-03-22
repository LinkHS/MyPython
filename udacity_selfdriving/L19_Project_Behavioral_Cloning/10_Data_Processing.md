**see the video**

## Lambda Layers

In Keras, [lambda layers](https://keras.io/layers/core/#lambda) can be used to create arbitrary functions that operate on each image as it passes through the layer.

In this project, a lambda layer is a convenient way to parallelize image normalization. The lambda layer will also ensure that the model will normalize input images when making predictions in `drive.py`. 

That lambda layer could take each pixel in an image and run it through the formulas:

```
pixel_normalized = pixel / 255
pixel_mean_centered = pixel_normalized - 0.5
```

A lambda layer will look something like:

```
Lambda(lambda x: (x / 255.0) - 0.5)
```

Below is some example code for how a lambda layer can be used. 

```python
from keras.models import Sequential, Model
from keras.layers import Lambda

# set up lambda layer
model = Sequential()
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
...
```

