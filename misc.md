# 字典 Dict

## `get` 和 `[]` 获取 key 的区别

```
result = {'comment1': 'OK'}

# 此时都返回 'OK'
result.get{'comment1'} == result['comment1']
>>>
True

# 返回 None
result.get{'comment2'}

# 程序出错
result.[comment2]
```



# Others

---
## 获取 for 循环的当前步数 
Q: Accessing the index in 'for' loops? 
Q: Using a for loop, how do I access the loop index?

```
for index, item in enumerate(items):
    print(index, item)

# ---
for count, item in enumerate(items, start=1):
    print(count, item)
```

---
## tuple 元祖索引
`b[x:y:z]` 相当于 `b[start​:end:​step]`，x默认0，y默认-1，z默认1  
关于溢出，如果第一次递进超过end，就算溢出

```
# b = np.zero((1,3,2,4)).shape
b = (1, 3, 2, 4) # 等价于 np.zero((1,3,2,4)).shape
print(b)
print('0:', b[0:0:1])   # 从0开始，到0之前结束，递进1
print('1:', b[0:0:2])   # 从0开始，到0之前结束，递进2
print('2:', b[0:1:2])   # 从0开始，到1之前结束，递进2（溢出）
>>>
0: ()
1: ()
2: (1,)

# ---
print('3:', b[0:-1:2])  # 从0开始，到-1（最后一个）之前结束，递进2
print('4:', b[0::-1])   # 从0开始，递进-1（溢出）
print('5:', b[1::1])    # 从1开始，递进1
print('6:', b[1::-1])   # 从1开始，递进-1
>>>
3: (1, 2)
4: (1,)
5: (3, 2, 4)
6: (3, 1)

# ---
print('7:', b[2::1])    # 从2开始，递进1
print('8:', b[2::-1])   # 从2开始，递进-1
print('9:', b[2::2])    # 从2开始，递进2（溢出）
print('10:', b[1::2])   # 从1开始，递进2
>>>
7: (2, 4)
8: (2, 3, 1)
9: (2,)
10: (3, 4)
    
# ---
print('11:', b[0:])     # 等于b[0::]；从0开始
print('12:', b[0:2])    # 从0开始，到2之前结束
print('13:', b[0:1])    # 从0开始，到1之前结束（溢出，因为默认递进1吧）
print('14:', b[0:3])    # 从0开始，到3之前结束
>>>
11: (1, 3, 2, 4) (1, 3, 2, 4)
12: (1, 3)
13: (1,)
14: (1, 3, 2)

# ---
print('15:', b[0:3:-1]) # 从0开始，到3之前结束，递进-1
print('16:', b[0:3:3])  # 从0开始，到3之前结束，递进3（溢出）
print('17:', b[0:5])    # 从0开始，到5之前结束
print('18:', b[0:3:2])  # 从0开始，到3之前结束，递进2
>>>
15: ()
16: (1,)
17: (1, 3, 2, 4)
18: (1, 2)
```

---
## 自定义排序`sorted`
```
students = [['john', 'A', 15],
            ['jane', 'B', 12],
            ['dave', 'B', 10]]
sorted(students, key=lambda student: student[2]) 
>>>
[['dave', 'B', 10], ['jane', 'B', 12], ['john', 'A', 15]]
```

---
## 左移或右移一维 list/np.ndarray 中所有元素（循环）

> [English Writing] We can see that we correctly shifted all the values one position to the right,
wrapping up from the end of the array back to the begining.

```
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

    for i in range(n):
      dst[i] = src[(i-move) % n]
    
    return dst
```

---
## 结构体`namedtuple/__repr__`
```python
from collections import namedtuple
gaussian = namedtuple('Gaussian', ['mean', 'var'])
gaussian.__repr__ = lambda s: 'Ν(μ={:.3f}, σ={:.3f})'.format(s[0], s[1])
```

---

## `with` Statement
自动执行 `__enter__` 和 `__exit___`，可以用 `contextmanager` decorator 来创造

```python
from contextlib import contexmanager

@contextmanager
```

----

# CSV

## CSV 和 namedtuple 结合
```python
EmployeeRecord = namedtuple('EmployeeRecord', 'name, age, title, department, paygrade')

import csv
for emp in map(EmployeeRecord._make, csv.reader(open("employees.csv", "rb"))):
    print emp.name, emp.title
```

---
# argparse 

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
