```{.python .input}
# Quiz Solution
import tensorflow as tf

def run():
  output = None
  x = tf.placeholder(tf.int32)
 
  with tf.Session() as sess:
    output = sess.run(x, feed_dict={x: 123})
 
  return output
 
run()
```
