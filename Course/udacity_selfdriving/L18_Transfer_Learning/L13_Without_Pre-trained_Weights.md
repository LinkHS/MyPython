## Using Keras Applications models without pre-trained weights

So far, you've seen the effectiveness of models pre-trained on ImageNet weights, but what if we specify `weights=None` when we load a model? Well, you'll instead be randomly initializing the weights, as if you had built a model on your own and were starting from scratch.

From our chart before, there are few situations where this might even be a potential use case - basically, when you have data that is very different from the original data. However, given the large size of the ImageNet dataset (remember, it's over 14 million images from 1,000 classes!), it's highly unlikely this is really the case - it will almost always make the most sense to start with ImageNet pre-trained weights, and only fine-tune from there

Below, let's check out what happens when we try to use a pre-made model but set the weights to `None` - this means no training has occurred yet!

In the following lab, you'll get a chance to actually add layers to the end of a pre-trained model, so that you can actually use the full power of transfer learning, instead of just using it toward the 1,000 ImageNet classes as a whole.

