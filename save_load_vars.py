## Saving or loading multiple variables:
```
import pickle
import numpy as np

obj1 = np.eye(2)
obj2 = [1]
fn = 'test.pkl'

# Saving the objects
with open(fn, 'wb') as f: # Python 3: open(..., 'wb')
    pickle.dump([obj1, obj2], f)

# Getting back the objects:
with open(fn, 'rb') as f:  # Python 3: open(..., 'rb')
    obj1, obj2 = pickle.load(f)
```
