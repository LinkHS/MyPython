
# Solutions
## Problem 1
Implement the Min-Max scaling function ($X'=a+{\frac {\left(X-X_{\min }\right)\left(b-a\right)}{X_{\max }-X_{\min }}}$) with the parameters:

$X_{\min }=0$

$X_{\max }=255$

$a=0.1$

$b=0.9$


```python
# Problem 1 - Implement Min-Max scaling for grayscale image data
def normalize_grayscale(image_data):
    """
    Normalize the image data with Min-Max scaling to a range of [0.1, 0.9]
    :param image_data: The image data to be normalized
    :return: Normalized image data
    """
    a = 0.1
    b = 0.9
    grayscale_min = 0
    grayscale_max = 255
    return a + ( ( (image_data - grayscale_min)*(b - a) )/( grayscale_max - grayscale_min ) )
```

## Problem 2
- Use [tf.placeholder()](https://www.tensorflow.org/api_docs/python/io_ops.html#placeholder) for `features` and `labels` since they are the inputs to the model.
- Any math operations must have the same type on both sides of the operator.  The weights are float32, so the `features` and `labels` must also be float32.
- Use [tf.Variable()](https://www.tensorflow.org/api_docs/python/state_ops.html#Variable) to allow `weights` and `biases` to be modified.
- The `weights` must be the dimensions of features by labels.  The number of features is the size of the image, 28*28=784.  The size of labels is 10.
- The `biases` must be the dimensions of the labels, which is 10.


```python
features_count = 784
labels_count = 10

# Problem 2 - Set the features and labels tensors
features = tf.placeholder(tf.float32)
labels = tf.placeholder(tf.float32)

# Problem 2 - Set the weights and biases tensors
weights = tf.Variable(tf.truncated_normal((features_count, labels_count)))
biases = tf.Variable(tf.zeros(labels_count))
```

# Problem 3
Configuration 1
* **Epochs:** 1
* **Batch Size:** 50
* **Learning Rate:** 0.01

Configuration 2
* **Epochs:** 1
* **Batch Size:** 100
* **Learning Rate:** 0.1

Configuration 3
* **Epochs:** 4 or 5
* **Batch Size:** 100
* **Learning Rate:** 0.2
