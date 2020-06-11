## Introduction
For the next few quizzes we'll test your understanding of the dimensions in CNNs. Understanding dimensions will help you make accurate tradeoffs between model size and performance. As you'll see, some parameters have a much bigger impact on model size than others.

---
## Setup
H = height, W = width, D = depth
- We have an input of shape 32x32x3 (HxWxD)
- 20 filters of shape 8x8x3 (HxWxD)
- A stride of 2 for both the height and width (S)
- With padding of size 1 (P)

Recall the formula for calculating the new height or width:

`new_height = (input_height - filter_height + 2 * P)/S + 1`  
`new_width = (input_width - filter_width + 2 * P)/S + 1`  

---
## Quiz:
Convolutional Layer Output Shape  
What's the shape of the output?
> The answer format is HxWxD, so if you think the new height is 9, new width is 9, and new depth is 5, then type 9x9x5.

**Ans**  
14x14x20
