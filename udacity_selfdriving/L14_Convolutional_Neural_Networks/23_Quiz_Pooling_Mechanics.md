## Setup
H = height, W = width, D = depth
- We have an input of shape 4x4x5 (HxWxD)
- Filter of shape 2x2 (HxW)
- A stride of 2 for both the height and width (S)

Recall the formula for calculating the new height or width:  

`new_height = (input_height - filter_height)/S + 1`  
`new_width = (input_width - filter_width)/S + 1`

NOTE: For a pooling layer the output depth is the same as the input depth. Additionally, the pooling operation is applied individually for each depth slice.

The image below gives an example of how a max pooling layer works. In this case, the max pooling filter has a shape of 2x2. As the max pooling filter slides across the input layer, the filter will output the maximum value of the 2x2 square.

![image](../data/L14_23.png)

**Quiz**  
Pooling Layer Output Shape  
What's the shape of the output? Format is HxWxD.

