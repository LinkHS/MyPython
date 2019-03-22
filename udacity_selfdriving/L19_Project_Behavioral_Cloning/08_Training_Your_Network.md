Note that the workspace for this project is equipped with a GPU which can be enabled for training your network. So, rather than working in a shell on AWS you can enable GPU mode and work in a workspace shell.

**see the video**

> NOTE: `cv2.imread` will get images in BGR format, while `drive.py` uses RGB. In the video above one way you could keep the same image formatting is to do `image = ndimage.imread(current_path)` with `from scipy import ndimage` instead.

#### Training Your Network

Now that you have training data, it’s time to build and train your network!

Use Keras to train a network to do the following:

1. Take in an image from the center camera of the car. This is the input to your neural network.
2. Output a new steering angle for the car.

You don’t have to worry about the throttle for this project, that will be set for you.

[Save your trained model](https://keras.io/getting-started/faq/#how-can-i-save-a-keras-model) architecture as `model.h5` using model.save('model.h5').

