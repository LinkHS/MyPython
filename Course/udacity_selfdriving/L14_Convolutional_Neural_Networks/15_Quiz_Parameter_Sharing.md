Now we'd like you to calculate the number of parameters in the convolutional layer, if every neuron in the output layer shares its parameters with every other neuron in its same channel.

This is the number of parameters actually used in a convolution layer (`tf.nn.conv2d()`).

---
## Setup
H = height, W = width, D = depth
- We have an input of shape 32x32x3 (HxWxD)
- 20 filters of shape 8x8x3 (HxWxD)
- A stride of 2 for both the height and width (S)
- Zero padding of size 1 (P)

Output Layer
- 14x14x20 (HxWxD)

**Hint**    
With parameter sharing, each neuron in an output channel shares its weights with every other neuron in that channel. So the number of parameters is equal to the number of neurons in the filter, plus a bias neuron, all multiplied by the number of channels in the output layer.

**Quiz: Convolution Layer Parameters 2**  
How many parameters does the convolution layer have (with parameter sharing)?

That's right, there are 3860 total parameters. That's 196 times fewer parameters! Here's how the answer is calculated:

(8 * 8 * 3 + 1) * 20 = 3840 + 20 = 3860

That's 3840 weights and 20 biases. This should look similar to the answer from the previous quiz. The difference being it's just 20 instead of (14 * 14 * 20). Remember, with weight sharing we use the same filter for an entire depth slice. Because of this we can get rid of 14 * 14 and be left with only 20.
