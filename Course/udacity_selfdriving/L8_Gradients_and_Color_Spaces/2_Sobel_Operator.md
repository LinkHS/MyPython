# Sobel Operator

The Sobel operator is at the heart of the Canny edge detection algorithm you used in the Introductory Lesson. Applying the Sobel operator to an image is a way of taking the derivative of the image in the $x$ or $y$ direction. The operators for $Sobel_x$ and $Sobel_y$, respectively, look like this:

![image](../data/L8_2.png)

These are examples of Sobel operators with a kernel size of 3 (implying a 3 x 3 operator in each case). This is the minimum size, but the kernel size can be any odd number. A larger kernel implies taking the gradient over a larger region of the image, or, in other words, a smoother gradient.

To understand how these operators take the derivative, you can think of overlaying either one on a 3 x 3 region of an image. If the image is flat across that region (i.e., there is little change in values across the given region), then the result (summing the element-wise product of the operator and corresponding image pixels) will be zero.

$$gradient - \sum(region*S_x)$$

For example, given:
$$region = \left( \begin{array} { c c c } 2 & 2 & 2 \\ 2 & 2 & 2 \\ 2 & 2 & 2 \end{array} \right) , S_x = \left(\begin{array} { c c c } -1 & 0 & 1 \\ -2 & 0 & 2 \\ -1 & 0 & 1 \end{array} \right)$$

The element-wise product would be:
$$\left(\begin{array} {c c c} -2 & 0 & 2 \\ -4 & 0 & 4 \\ -2 & 0 & 2 \end{array} \right)$$

In which case, the sum of this matrix is 0, implying a flat gradient (in the x-direction in this calculation, although the y-direction is also zero in this example).

If, instead, for example, you apply the $S_x$ operator to a region of the image where values are rising from left to right, then the result will positive, implying a positive derivative.

Given:
$$region = \left(\begin{array} {c c c} 1 & 2 & 3 \\ 1 & 2 & 3 \\ 1 & 2 & 3 \end{array}\right), S_x = \left(\begin{array} { c c c } -1 & 0 & 1 \\ -2 & 0 & 2 \\ -1 & 0 & 1 \end{array} \right)$$

The element-wise product would be:
$$\left(\begin{array} {c c c} -1 & 0 & 3 \\ -1 & 0 & 6 \\ -1 & 0 & 3 \end{array} \right)$$

This time ,the sum of this matrix is 8, meaning a gradient exists in the x-direction. Note that in this example image region, if you applied the $S_y$ operator, the result would be a gradient of 0 in the y-direction, as the values are not varying from top to bottom.

## Visual Example
If we apply the Sobel $x$ and $y$ operators to this image:
![image](../data/L8_2.jpg)

And then we take the absolute value, we get the result:
![image](../data/L8_2_1.png)
Absolute value of Sobel x (left) and Sobel y (right).

**x vs. y**
In the above images, you can see that the gradients taken in both the $x$ and the $y$ directions detect the lane lines and pick up other edges. Taking the gradient in the $x$ direction emphasizes edges closer to vertical. Alternatively, taking the gradient in the $y$ direction emphasizes edges closer to horizontal.

# To Finish
