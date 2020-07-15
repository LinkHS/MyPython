# 第6章 Python数据结构


## `namedtuple`

```python
from collections import namedtuple

Stock = namedtuple("Stock", "symbol current high low")
stock = Stock("FB", 75.00, high=75.03, low=74.90)

stock
```

## `defaultdict`

```python
from collections import defaultdict

def letter_frequency(sentence):
    frequencies = defaultdict(int)
    for letter in sentence:
        frequencies[letter] += 1
    return frequencies

letter_frequency('hello')
```

## `Counter`

```python
from collections import Counter

def letter_frequency(sentence):
    return Counter(sentence)

letter_frequency('hello')
```
