import numpy as np
import tensorflow as tf
from tensorflow.python.framework.errors import FailedPreconditionError
import os
import sys

def is_weights_good(w):
    w_answer = [[-0.01811021,  0.51838213],
 [ 0.05832403, -0.48847285],
 [-0.37598562, -0.7711397 ],
 [-0.5922465, -0.3118519 ],
 [ 0.21055079, -1.1010232 ]] 
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        w_result = sess.run(w)
      
    return np.allclose(w_answer, w_result)


def is_biases_good(b):
    b_answer = [0.0, 0.0]
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        b_result = sess.run(b)
        
    return np.array_equal(b_answer, b_result)


def is_linear_good(l, test_input):
    
    logits_answer = [[-2.34565091, -9.52450562],[-8.03849602, -9.28480148]]

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        logits_result = sess.run(l, feed_dict={test_input: [[1,2,3,4,5], [6,7,8,9,0]]})
        
    return np.allclose(logits_answer, logits_result)

def get_result(get_weights, get_biases, linear):
    result = {
        'correct': False,
        'feedback': 'That\'s the wrong answer.',
        'comment': ''}

    tf.set_random_seed(123456)
    
    n_features = 5
    n_labels = 2
    test_input = tf.placeholder(tf.float32)
    
    weights = get_weights(n_features, n_labels)
    biases = get_biases(n_labels)
    lin = linear(test_input, weights, biases)

    if not isinstance(weights, tf.Variable):
        result['feedback'] = 'Function weights not returning tf.Variable type.'
        result['comment'] = 'Use the tf.Variable function.'
    elif not isinstance(biases, tf.Variable):
        result['feedback'] = 'Function biases not returning tf.Variable type.'
        result['comment'] = 'Use the tf.Variable function.'
    elif weights.get_shape() != (n_features, n_labels):
        result['feedback'] = 'Function weights is returning the wrong shape.'
    elif biases.get_shape() != n_labels:
        result['feedback'] = 'Function biases is returning the wrong shape.'
    elif not is_weights_good(weights):
        result['feedback'] = 'Function weights isn\'t correct.'
    elif not is_biases_good(biases):
        result['feedback'] = 'Function biases isn\'t correct.'
    elif not is_linear_good(lin, test_input):
        import pdb;pdb.set_trace()
        result['feedback'] = 'Function linear isn\'t correct.'
    else:
        try:
            std_out = sys.stdout
            f = open(os.devnull, 'w')
            sys.stdout = f
        
        except FailedPreconditionError as err:
            if err.message.startswith('Attempting to use uninitialized value Variable'):
                result['feedback'] = 'At least one variable is not initialized.'
            else:
                raise err
        else:
            result['correct'] = True
            result['feedback'] = 'You got it!  That\'s the correct answer.'
        finally:
            sys.stdout = std_out
    return result

def run_grader(get_weights, get_biases, linear):
    
    try:
    # Get grade result information
        result = get_result(get_weights, get_biases, linear)
    except Exception as err:
        # Default error result
        result = {
            'correct': False,
            'feedback': 'Something went wrong with your submission:',
            'comment': str(err)}

    feedback = result.get('feedback')
    comment = result.get('comment')

    print(f"{feedback}\n{comment}\n")



if __name__ == "__main__":
    run_grader(get_weights, get_biases, linear)