# np.array 操作

## - 生成数据、初始化
```python
"""快速浮点
If the list contains all ints then the created array will also have a data type of int, otherwise it will befloat.
"""
np.array([1., 0, 0, 0, 0, 0])

"""随机数
""" 
# 创建2行2列取值范围为[0,1)的数组
np.random.rand(2,2)

# 创建2行3列，取值范围为标准正态分布的数组
arr2 = np.random.randn(2,3)

# 对角矩阵
np.diag([500., 49.])
>>>
array([[500.,   0.],
       [  0.,  49.]])
```



## - 矩阵相乘，`np.matmul(A,B)`

请注意W.shape
```python
W = np.arange(2)
X = np.arange(6).reshape(3, 2)
b = 1

print(X.shape, W.shape, '\n')
print(np.matmul(W, X.T) + b, '\n')
print(np.matmul(X, W) + b, '\n')

>>>
(3, 2) (2,) 

[2 4 6] 

[2 4 6]
```

## - 添加
`np.append`

```python
a = np.array([1,2,3])
np.append(a, 3)
>>>
array([1, 2, 3, 3])
```

### 001. Append a NumPy array to a NumPy array
```python
a = np.array([[1,2,3],[2,3,4]])
b = np.array([[1],[2]])

np.append(a, b, axis=1)
>>>
array([[1, 2, 3, 1],
       [2, 3, 4, 2]])
```

### 002. 二维的添加，等价与下面的`np.hstack`
```python
a = np.array([[0,0,0], [1,1,1]])
b = np.append(a, np.zeros((a.shape[0], 1), dtype=a.dtype), axis=1)
b
>>>
array([[0, 0, 0, 0],
       [1, 1, 1, 0]])
```

`np.hstack`
```python
a = np.array([[0,0,0], [1,1,1]])
b = np.hstack((a, np.zeros((a.shape[0], 1), dtype=a.dtype)))
b
>>>
array([[0, 0, 0, 0],
       [1, 1, 1, 0]])
```

## - 删除
```python
a = [1,2,3,4]
a.pop(-1)
a
>>>
[1, 2, 3]
```

## - 位置操作
```python
a = np.array([1,1,2,3])
a = a[(a==1) | (a==2)]
a
>>>
array([1, 1, 2])
```

---

# Misc

## 01 Assign all colums of a matrix to variables
```python
a = np.array([[1,2,3], [1,2,3]])
x1, x2, x3 = a.transpose()
x1, x2, x3
>>>
(array([1, 1]), array([2, 2]), array([3, 3]))
```

---
## 02 避免使用`for`循环
```python
A = np.array([1,2,3,1,1])
A[A==1] = 2
A

> array([2, 2, 3, 2, 2])
```

---
## 03 注意 np.array 即使size, shape相同，但是可能dtype不一样
https://www.numpy.org/devdocs/reference/arrays.ndarray.html#constructing-arrays

```python
# 注意此时 A.dtype 为 uint8
A = matplotlib.image.imread('pic.jpg')
B = np.zeros_like(A)

# 此时A和B的dtype不一样
B = np.zeros(A.shape)

# 改成
B = np.zeros(A.shape, dtype=A.dtype)
```

---
## 04 打印精度设置, `np.set_printoptions()`
```python
np.set_printoptions(2)

np.set_printoptions(formatter={'float': '{:3.2d}'.format})

np.set_printoptions(formatter={'int_kind': '{:2d}'.format}) # 整数占2个字符，不足补空格
```

---
## 05 `np.arctan2` 弧度、角度计算
可以有效避免分母为0(90°)的情况
```python
np.arctan2(-10, 0.1) * 180 / np.pi
>>>
-89.42
```

