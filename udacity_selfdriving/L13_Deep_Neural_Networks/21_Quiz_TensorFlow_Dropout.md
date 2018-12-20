![image](../data/L13_21.jpeg)
Figure 1: Taken from the paper "Dropout: A Simple Way to Prevent Neural Networks from

Dropout is a regularization technique for reducing overfitting. The technique temporarily drops units (artificial neurons) from the network, along with all of those units' incoming and outgoing connections. Figure 1 illustrates how dropout works.

TensorFlow provides the `tf.nn.dropout()` function, which you can use to implement dropout.

Let's look at an example of how to use `tf.nn.dropout()`.
 ```python
keep_prob = tf.placeholder(tf.float32) # probability to keep units
#
hidden_layer = tf.add(tf.matmul(features, weights[0]), biases[0])
hidden_layer = tf.nn.relu(hidden_layer)
hidden_layer = tf.nn.dropout(hidden_layer, keep_prob)
#
logits = tf.add(tf.matmul(hidden_layer, weights[1]), biases[1])
 ```

The code above illustrates how to apply dropout to a neural network.

The `tf.nn.dropout()` function takes in two parameters:
- hidden_layer: the tensor to which you would like to apply dropout

- keep_prob: the probability of keeping (i.e. not dropping) any given unit

`keep_prob` allows you to adjust the number of units to drop. In order to compensate for dropped units, `tf.nn.dropout()` multiplies all units that are kept (i.e. not dropped) by `1/keep_prob`.

During training, a good starting value for `keep_prob` is 0.5.

During testing, **use a `keep_prob value` of 1.0** to keep all units and maximize the power of the model.


Quiz 1

Take a look at the code snippet below. Do you see what's wrong?

There's nothing wrong with the syntax, however the test accuracy is extremely low.

 ```python
...
#
keep_prob = tf.placeholder(tf.float32) # probability to keep units
#
hidden_layer = tf.add(tf.matmul(features, weights[0]), biases[0])
hidden_layer = tf.nn.relu(hidden_layer)
hidden_layer = tf.nn.dropout(hidden_layer, keep_prob)
#
logits = tf.add(tf.matmul(hidden_layer, weights[1]), biases[1])
#
...
#
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    #
    for epoch_i in range(epochs):
        for batch_i in range(batches):
            ....
            sess.run(optimizer, feed_dict={
                features: batch_features,
                labels: batch_labels,
                keep_prob: 0.5})
    #
    validation_accuracy = sess.run(accuracy, feed_dict={
        features: test_features,
        labels: test_labels,
        keep_prob: 0.5})
 ```

**Quiz**  
What's wrong with the above code?
1. Dropout doesn't work with batching.
2. The keep_prob value of 0.5 is too low.
3. There shouldn't be a value passed to keep_prob when testing for accuracy.
4. keep_prob should be set to 1.0 when evaluating validation accuracy.


**Ans**  
4
> That's correct! You should only drop units while training the model. During validation or testing, you should keep all of the units to maximize accuracy.
