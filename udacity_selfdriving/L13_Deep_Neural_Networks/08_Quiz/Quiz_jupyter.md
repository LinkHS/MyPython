```{.python .input}
# Solution is available in the other "solution.py" tab
import tensorflow as tf

output = None
hidden_layer_weights = [
    [0.1, 0.2, 0.4],
    [0.4, 0.6, 0.6],
    [0.5, 0.9, 0.1],
    [0.8, 0.2, 0.8]]
out_weights = [
    [0.1, 0.6],
    [0.2, 0.1],
    [0.7, 0.9]]
```

```{.python .input}
# Weights and biases
weights = [
    tf.Variable(hidden_layer_weights),
    tf.Variable(out_weights)]
biases = [
    tf.Variable(tf.zeros(3)),
    tf.Variable(tf.zeros(2))]
```

```{.python .input}
# Input
features = tf.Variable([[1.0, 2.0, 3.0, 4.0], [-1.0, -2.0, -3.0, -4.0], [11.0, 12.0, 13.0, 14.0]])
```

```{.python .input}
# TODO: Create Model
# Hidden Layer with ReLU activation function
```

```{.python .input}
# TODO: save and print session results on a variable named "output"
```

The output should be:
 ```
 array([[  5.11000013,   8.44000053],
       [  0.        ,   0.        ],
       [ 24.01000214,  38.23999786]], dtype=float32)
 ```

```{.python .input}
### DON'T MODIFY ANYTHING BELOW ###
### Be sure to run all cells above before running this cell ###
import grader

try:
    grader.run_grader(output)
except Exception as err:
    print(str(err))
```
