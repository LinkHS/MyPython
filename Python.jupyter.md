# Python


## `random`

```python
import random

random.seed(0)
```

```python
# 随机生成[-10, 10] 之间的浮点数
random.uniform(-10, 10)
```

```python
# Return random integer in range [a, b], including both end points.
random.randint(-10, 10)
```

## 实用技巧

### `eval()`
`eval()`可以将输入的字符串转化为Python代码

```python
eval('print("hello")')
```

有时可以配合`input()`使用，如`eval(intput())`

```python
import numpy as np

eval('1+np.pi')
```

### `f`直接打印变量

```python
a = 'hello'
b = 2
print(f'{a}, {b}')
```

## System

```python
import os
import subprocess
```

### `os.path`

```python
os.path.basename("/home/austin/name.md.txt")
```
```python
# Getting the name of the file without the extension
os.path.splitext("/home/austin/name.md.txt")[0]
```
```python
# Get the extension
os.path.splitext("/home/austin/name.md.txt")[1]
```
```python
os.path.splitext(os.path.basename("/home/austin/name.md"))[0]
```
### `subprocess`

#### 得到系统内存 get_mem

```python
# Only for Linux or Mac
def get_mem():
    res = subprocess.check_output(['ps', 'u', '-p', str(os.getpid())])
    return int(str(res).split()[15]) / 1e3

get_mem()
```

## ipython
### `display`
如果需要显示`np.array`，请查看[`sympy.jupyter.md`](./sympy.jupyter.md)
```python
import numpy as np
from  IPython.display import display, Math, Latex
```

```python
raw_latex = "\\text{%s} \quad W*X^T" % ("e.g.")

raw_latex
```

```python
display(Math(raw_latex))
```

```python
display(Latex("$"+raw_latex+"$"))
```
