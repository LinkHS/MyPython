```{.python .input}
import tensorflow as tf
from grader import get_result

import tensorflow as tf

def run():
    output = None
    x = tf.placeholder(tf.int32)

    with tf.Session() as sess:
        output = sess.run(x, feed_dict={x: 123})

    return output
```

```{.python .input}
### DON'T MODIFY ANYTHING BELOW ###
### Be sure to run all cells above before running this cell ###
import grader

try:
    grader.get_result(run)
except Exception as err:
    print(str(err))
```
