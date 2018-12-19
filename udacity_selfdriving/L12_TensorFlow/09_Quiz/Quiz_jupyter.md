```{.python .input  n=1}
# Solution is available in the "solution.ipynb" 
import tensorflow as tf

# TODO: Convert the following to TensorFlow:
x = tf.constant(10)
y = tf.constant(2)
z = tf.subtract(tf.divide(x, y), tf.cast(tf.constant(1), tf.float64))

# TODO: Print z from a session as the variable output
with tf.Session() as sess:
    # Run the tf.constant operation in the session
    output = sess.run(z)
    print(output)
```

注意，python3.6 以下的版本不支持 `print(f"{var}")` 模式

```{.python .input  n=2}
### DON'T MODIFY ANYTHING BELOW ###
### Be sure to run all cells above before running this cell ###
import grader

try:
    grader.run_grader(output)
except Exception as err:
    print(str(err))
```
