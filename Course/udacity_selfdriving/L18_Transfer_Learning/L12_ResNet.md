see the video

[Here](https://arxiv.org/pdf/1512.03385.pdf) is the original ResNet paper, for those interested.

## ResNet in Keras

As you may have guessed, ResNet is also a model included in [Keras Applications](https://keras.io/applications/), under `ResNet50`.

```python
from keras.applications.resnet50 import ResNet50

model = ResNet50(weights='imagenet', include_top=False)
```

Again, you'll need to do ImageNet-related pre-processing if you want to use the pre-trained weights for it. ResNet50 has a 224x224 input size.
