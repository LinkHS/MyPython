We're now going to calculate the number of parameters of the convolutional layer. The answer from the last quiz will come into play here!

Being able to calculate the number of parameters in a neural network is useful since we want to have control over how much memory a neural network uses.

---
## Setup
H = height, W = width, D = depth
- We have an input of shape 32x32x3 (HxWxD)
- 20 filters of shape 8x8x3 (HxWxD)
- A stride of 2 for both the height and width (S)
- Zero padding of size 1 (P)

---
## Output Layer
- 14x14x20 (HxWxD)

---
## Hint
Without parameter sharing, each neuron in the output layer must connect to each neuron in the filter. In addition, each neuron in the output layer must also connect to a single bias neuron.

**Convolution Layer Parameters 1**  
How many parameters does the convolutional layer have (without parameter sharing)?

**Ans**  
756560

There are 756560 total parameters. That's a HUGE amount! Here's how we calculate it:

(8 * 8 * 3 + 1) * (14 * 14 * 20) = 756560

8 * 8 * 3 is the number of weights, we add 1 for the bias. Remember, each weight is assigned to every single part of the output (14 * 14 * 20). So we multiply these two numbers together and we get the final answer.
