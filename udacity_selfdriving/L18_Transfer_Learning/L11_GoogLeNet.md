see the video

You can find the original GoogLeNet/Inception paper [here](https://arxiv.org/pdf/1409.4842.pdf).

## GoogLeNet/Inception in Keras

Inception is also one of the models included in [Keras Applications](https://keras.io/applications/). Utilizing this model follows pretty much the same steps as using VGG, although this time you'll use the `InceptionV3` architecture.

```python
from keras.applications.inception_v3 import InceptionV3

model = InceptionV3(weights='imagenet', include_top=False)
```

Don't forget to perform the necessary pre-processing steps to any inputs you include! While the original Inception model used a 224x224 input like VGG, InceptionV3 actually uses a 299x299 input.
