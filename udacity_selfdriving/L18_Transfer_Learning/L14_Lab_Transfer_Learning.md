## Lab: Transfer Learning

In the below lab, you'll get a chance to try out a few instances of transfer learning, including both frozen and non-frozen pre-trained weights.

#### Frozen Weights

Frozen weights are often used when only fine-tuning the model, as backpropagation and weight updates will not be applied to any frozen layers during training. If you have an ImageNet pre-trained model, most of the network is likely applicable to your situation, so you may only need to cut off the top fully-connected layer, freeze all other layers, and just add one or more layers at the end that are not frozen to perform some fine-tuning.

There is also the option of not freezing the weights, which will start your model on the ImageNet pre-trained weights (if applicable) and then perform further training from there. 

An additional benefit of freezing the weights also comes in the form of memory usage and training speed - for the larger networks such as VGG, there is a substantially larger memory usage and slower speed when it needs to perform backpropagation and weight updates across all layers instead of just on a small portion of (likely smaller) layers.

*Note*: There is a solution notebook that can be found by clicking on the Jupyter logo in the upper left of the workspace if you get stuck.

[Lab](./Lab_14_Transfer_learning.md)

