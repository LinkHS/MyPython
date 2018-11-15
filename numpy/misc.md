
---
## `np.arctan2` 弧度、角度计算
可以有效避免分母为0(90°)的情况
```
np.arctan2(-10, 0.1) * 180 / np.pi
>>>
-89.42
```

---
## np.array 操作
添加
`np.append`
```
a = np.array([1,2,3])
np.append(a, 3)
>>>
array([1, 2, 3, 3])
```

二维的添加，等价与下面的`np.hstack`
```
a = np.array([[0,0,0], [1,1,1]])
b = np.append(a, np.zeros((a.shape[0], 1), dtype=a.dtype), axis=1)
b
>>>
array([[0, 0, 0, 0],
       [1, 1, 1, 0]])
```

`np.hstack`
```
a = np.array([[0,0,0], [1,1,1]])
b = np.hstack((a, np.zeros((a.shape[0], 1), dtype=a.dtype)))
b
>>>
array([[0, 0, 0, 0],
       [1, 1, 1, 0]])
```


删除
```
a = [1,2,3,4]
a.pop(-1)
a
>>>
[1, 2, 3]
```

位置操作
```
a = np.array([1,1,2,3])
a = a[(a==1) | (a==2)]
a
>>>
array([1, 1, 2])
```

## Assign all colums of a matrix to variables
```
a = np.array([[1,2,3], [1,2,3]])
x1, x2, x3 = a.transpose()
x1, x2, x3
>>>
(array([1, 1]), array([2, 2]), array([3, 3]))
```

---
## Append a NumPy array to a NumPy array
```
a = np.array([[1,2,3],[2,3,4]])
b = np.array([[1],[2]])

np.append(a, b, axis=1)
>>>
array([[1, 2, 3, 1],
       [2, 3, 4, 2]])
```

---
## 避免使用`for`循环
```
A = np.array([1,2,3,1,1])
A[A==1] = 2
A

> array([2, 2, 3, 2, 2])
```

---
## 注意 np.array 即使size, shape相同，但是可能dtype不一样
https://www.numpy.org/devdocs/reference/arrays.ndarray.html#constructing-arrays

```
# 注意此时 A.dtype 为 uint8
A = matplotlib.image.imread('pic.jpg')
B = np.zeros_like(A)

# 此时A和B的dtype不一样
B = np.zeros(A.shape)

# 改成
B = np.zeros(A.shape, dtype=A.dtype)
```
