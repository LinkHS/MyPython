

```python
import pickle
import numpy as np
import tensorflow as tf

# Load pickled data
with open('../data/small_train_traffic.p', mode='rb') as f:
    data = pickle.load(f)
```


```python
# split data
X_train, y_train = data['features'], data['labels']
```


```python
# Setup Keras
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten
```


```python
# Build the Fully Connected Neural Network in Keras Here
model = Sequential()
model.add(Flatten(input_shape=(32, 32, 3)))
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dense(5))
model.add(Activation('softmax'))

# An Alternative Solution
# model = Sequential()
# model.add(Flatten(input_shape=(32, 32, 3)))
# model.add(Dense(128, activation='relu'))
# model.add(Dense(5, activation='softmax'))
```


```python
# preprocess data
X_normalized = np.array(X_train / 255.0 - 0.5 )

from sklearn.preprocessing import LabelBinarizer
label_binarizer = LabelBinarizer()
y_one_hot = label_binarizer.fit_transform(y_train)

model.compile('adam', 'categorical_crossentropy', ['accuracy'])
history = model.fit(X_normalized, y_one_hot, epochs=3, validation_split=0.2)
```
