# python


## 常用
### Dict and Easydict

#### `get` 和 `[]` 获取 key 的区别

```python
result = {'comment1': 'OK'}

# 此时都返回 'OK'
result.get('comment1') == result['comment1']
```

```python
# 返回 None
print(result.get('comment2'))
```

```python
# 程序出错
try:
    result['comment2']
except Exception as E:
    print("Exception: {}".format(type(E).__name__))
    print("Exception message: {}".format(E))
```

<!-- #region -->
#### Easydict
```python
from easydict import EasyDict as edict

__C = edict()
cfg = __C

__C.DATASET                             = edict()
__C.DATASET.NAME                        = 'GoPro'
```
<!-- #endregion -->

### 索引
`b[start:end:step]`

```python
b = (1, 3, 2, 4)

print('0:', b[0:4:1])   # 从0开始，到4之前结束，递进1
print('2:', b[0:-1:2])   # 从0开始，到最后一个结束，递进2
```

### Random

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

### namedtuple

```python
from collections import namedtuple
gaussian = namedtuple('Gaussian', ['mean', 'var'])
gaussian.__repr__ = lambda s: 'Ν(μ={:.3f}, σ={:.3f})'.format(s[0], s[1])

g = gaussian(0, 1)
g, g.mean, g.var
```

#### CSV 和 namedtuple 结合

```python
import csv

EmployeeRecord = namedtuple('EmployeeRecord', 'name, age')

csv_data = open("_files/employee.csv", "r")

for emp in map(EmployeeRecord._make, csv.reader(csv_data)):
    print(emp.name, emp.age)
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

### Deep Copy

```python
import copy

A = [[1], [2]]

B = copy.copy(A)
C = copy.deepcopy(A)

A[0][0] = ['love']

print('Shallow copy:', B, "\nDeep copy:", C)
```

### Print out nicely - `pprint`
尤其对字典和json文件有用

```python
import pprint

students = {"Dilip": ["English", "Maths", "Science"],
            "Raju": {"English": 50, "Maths": 60, "Science": 70},
            "Kalpana": (50, 60, 70)}

pprint.pprint(students)

pp = pprint.PrettyPrinter(width=20)
pp.pprint(students)
```

### 获取 for 循环的当前步数 

Q: Accessing the index in 'for' loops? 
Q: Using a for loop, how do I access the loop index?

```python
for count, item in enumerate(range(4), start=10):
    print(count, item)
```

### 自定义排序`sorted`

```python
students = [['john', 'A', 15],
            ['jane', 'B', 12],
            ['dave', 'B', 10]]

sorted(students, key=lambda student: student[2]) 
```

### 循环移动一维 list/np.ndarray
We can see that we correctly shifted all the values one position to the right, wrapping up from the end of the array back to the begining.

```python
def shift_1d(src, move):
    """ move the position by `move` spaces, where positive is 
    to the right, and negative is to the left
    """
    n = len(src)
    if isinstance(src, list):
        dst = [0] * n
    elif isinstance(src, np.ndarray):
        dst = np.zeros(n)
    else:
        raise Exception

    # 不要用for循环，太耗时
    if move > 0:
        dst = src[n-move%n:] + src[:move%n+1]
    else:
        dst = src[abs(move)%n:] + src[:abs(move)%n]
    return dst

shift_1d([1,2,3,4, 5], 2), shift_1d([1,2,3,4, 5], -4) 
```

### `with` Statement
#### contextlib
自动执行 `__enter__` 和 `__exit___`，可以用 `contextmanager` decorator 来创造。

例如自动管理文件打开和关闭

```python
from contextlib import contextmanager

@contextmanager
def open_file(path, mode):
    """
        with open_file() as f:
            ...
    """
    the_file = open(path, mode)
    yield the_fil
    the_file.close()
```

自动加上html的tag

```python
@contextmanager
def tag(name):
    print("<%s>" % name)
    yield
    print("</%s>" % name)

with tag("h1"):
    print("foo")
```

#### `with` 和 time Benchmark

构造一个`Benchmark`类，配合`with`测量代码运行时间

```python
import time

class Benchmark():
    def __init__(self, prefix=None):
        self.prefix = prefix + ' ' if prefix else ''
    
    def __enter__(self):
        self.start = time.time()
        
    def __exit__(self, *args):
        print('%stime: %.4f sec' % (self.prefix, time.time() - self.start))
```

```python
with Benchmark('Workloads are queued.'):
    print('+++')

with Benchmark('Workloads are finished.'):
    print('---')
```

<!-- #region -->
### 保存变量
如果变量非常大，想重复运算，可以保存加快运行速度
```python
import pickle

obj1 = np.eye(2)
obj2 = [1]
fn = 'test.pkl'

# Saving the objects
with open(fn, 'wb') as f:
    pickle.dump([obj1, obj2], f)
    
# Getting back the objects:
with open(fn, 'rb') as f:
    obj1, obj2 = pickle.load(f)
```
<!-- #endregion -->

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
```python
os.path.dirname("/home/austin/name.md")
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

### `import`相对路径文件

```
DirA
|--main.py
|--DirB
|----fileB.py
|----fileB1.py
```

在"main.py"中可以直接`import DirB.fileB`，但是同级/上级调用如在"fileB.py"里`import fileB1`要分几种情况：

- `python main.py`中调用了`import DirB.fileB`然后间接调用了`import fileB1`
- `python DirB/fileB.py`中调用了`import fileB1`
- `python fileB.py`中调用了`import fileB1`

鉴于情况比较复杂，尽量避免同级/上级调用，如需要运行`python fileB.py`来执行`if __name__ == "__main__"`的入口测试可以先设置以下python绝对或者相对环境：
```
export PYTHONPATH="/home/austin/.../DirA/DirB"
```

<!-- #region -->
### argparse
```python
import argparse

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('integers', metavar='N', type=int, nargs='+',
                    help='an integer for the accumulator')
parser.add_argument('--sum', dest='accumulate', action='store_const',
                    const=sum, default=max,
                    help='sum the integers (default: find the max)')

args = parser.parse_args()
print(args.accumulate(args.integers))
```
<!-- #endregion -->

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
