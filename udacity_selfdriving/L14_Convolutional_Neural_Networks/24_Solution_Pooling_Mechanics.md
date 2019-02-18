The answer is 2x2x5. Here's how it's calculated using the formula:

`(4 - 2)/2 + 1 = 2`
`(4 - 2)/2 + 1 = 2`

The depth stays the same.

Here's the corresponding code:

 ```python
input = tf.placeholder(tf.float32, (None, 4, 4, 5))
filter_shape = [1, 2, 2, 1]
strides = [1, 2, 2, 1]
padding = 'VALID'
pool = tf.nn.max_pool(input, filter_shape, strides, padding)
 ```
The output shape of `pool` will be [1, 2, 2, 5], even if `padding` is changed to 'SAME'.
