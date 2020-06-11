import numpy as np
import json

def get_result(student_output):
    """
    Run unit tests against <student_func>
    """
    
    answer = 4.0
    result = {
        'correct': False,
        'feedback': 'That\'s the wrong answer.  It should print {}'.format(answer),
        'comment': ''}

    try:
        student_output = np.float32(student_output)
        if not student_output:
            result['feedback'] = 'No output found'
        elif student_output == answer:
            result['correct'] = True
            result['feedback'] = 'That\'s right!  You correctly turned the algorithm to TensorFlow'
    except TypeError as err:
        if str(err).endswith('into a Tensor or Operation.)'):
            result['feedback'] = 'TensorFlow session requires a tensor to run'
        else:
            raise

    return result


def run_grader(student_output):

    try:
        # Get grade result information
        result = get_result(student_output)
    except Exception as err:
        # Default error result
        result = {
            'correct': False,
            'feedback': 'Something went wrong with your submission:',
            'comment': str(err)}
        
    feedback = result.get('feedback')
    comment = result.get('comment')

    print(f"{feedback}\n {comment}\n")