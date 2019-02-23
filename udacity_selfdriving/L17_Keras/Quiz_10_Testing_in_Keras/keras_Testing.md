

```python
import pickle
import numpy as np
import tensorflow as tf

# Load pickled data
with open('../data/small_train_traffic.p', mode='rb') as f:
    data = pickle.load(f)
```


```python
# Split the data
X_train, y_train = data['features'], data['labels']
```


```python
# Setup Keras
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
```


```python
# TODO: Build the Final Test Neural Network in Keras Here
```


```python
# preprocess data
X_normalized = np.array(X_train / 255.0 - 0.5 )

from sklearn.preprocessing import LabelBinarizer
label_binarizer = LabelBinarizer()
y_one_hot = label_binarizer.fit_transform(y_train)
```


```python
# compile and fit the model
model.compile('adam', 'categorical_crossentropy', ['accuracy'])
history = model.fit(X_normalized, y_one_hot, epochs=3, validation_split=0.2)
```


```python
# evaluate model against the test data
with open('small_test_traffic.p', 'rb') as f:
    data_test = pickle.load(f)

X_test = data_test['features']
y_test = data_test['labels']

# preprocess data
X_normalized_test = np.array(X_test / 255.0 - 0.5 )
y_one_hot_test = label_binarizer.fit_transform(y_test)

print("Testing")

metrics = model.evaluate(X_normalized_test, y_one_hot_test)
for metric_i in range(len(model.metrics_names)):
    metric_name = model.metrics_names[metric_i]
    metric_value = metrics[metric_i]
    print('{}: {}'.format(metric_name, metric_value))    
```


```python
### DON'T MODIFY ANYTHING BELOW ###
### Be sure to run all cells above before running this cell ###
import grader

try:
    grader.run_grader(metrics)
except Exception as err:
    print(str(err))
```
