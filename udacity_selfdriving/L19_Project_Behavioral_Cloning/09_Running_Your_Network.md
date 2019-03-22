Note that this video describes the process for running your network on a local machine with a model built using an AWS GPU instance. Since the project is in a GPU enabled workspace there is no need to download a GitHub repo (we have included the repo in the workspace) or transfer the model (the model should be trained in the workspace with GPU enabled).

**see the video**

### Validating Your Network

In order to validate your network, you'll want to compare model performance on the training set and a validation set. The validation set should contain image and steering data that was not used for training. A rule of thumb could be to use 80% of your data for training and 20% for validation or 70% and 30%. Be sure to randomly shuffle the data before splitting into training and validation sets.

If model predictions are poor on both the training and validation set (for example, mean squared error is high on both), then this is evidence of underfitting. Possible solutions could be to 

- increase the number of epochs
- add more convolutions to the network.

When the model predicts well on the training set but poorly on the validation set (for example, low mean squared error for training set, high mean squared error for validation set), this is evidence of overfitting. If the model is overfitting, a few ideas could be to

- use dropout or pooling layers
- use fewer convolution or fewer fully connected layers
- collect more data or further augment the data set

Ideally, the model will make good predictions on both the training and validation sets. The implication is that when the network sees an image, it can successfully predict what angle was being driven at that moment. 

#### Testing Your Network

Once you're satisfied that the model is making good predictions on the training and validation sets, you can test your model by launching the simulator and entering autonomous mode. 

The car will just sit there until your Python server connects to it and provides it steering angles. Here’s how you start your Python server:

```
python drive.py model.h5
```

Once the model is up and running in `drive.py`, you should see the car move around (and hopefully not off) the track! If your model has low mean squared error on the training and validation sets but is driving off the track, this could be because of the data collection process. It's important to feed the network examples of good driving behavior so that the vehicle stays in the center and recovers when getting too close to the sides of the road. 

