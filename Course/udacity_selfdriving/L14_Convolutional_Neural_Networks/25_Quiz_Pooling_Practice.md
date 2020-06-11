Great, now let's practice doing some pooling operations manually.
Max Pooling

What's the result of a max pooling operation on the input:
 ```
[[[0, 1, 0.5, 10],
   [2, 2.5, 1, -8],
   [4, 0, 5, 6],
   [15, 1, 2, 3]]]
 ```
Assume the filter is 2x2 and the stride is 2 for both height and width. The output shape is 2x2x1.

The answering format will be 4 numbers, each separated by a comma, such as: `1,2,3,4`.

Work from the top left to the bottom right

