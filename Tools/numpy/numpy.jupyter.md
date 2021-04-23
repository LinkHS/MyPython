# Numpy

![](../_files/np.png)

```python
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display, Math, Latex
```

## `np.array`操作

### - 生成数据、初始化

#### 指定数据
```python
"""
If the list contains all ints then the created array will also have a data type of int, 
otherwise it will be float.
"""
np.array([1., 0, 0, 0, 0, 0])
```

```python
np.array(range(10))**2
```

```python
# 对角矩阵
np.diag([500., 49.])
```

#### 随机数据

```python
# 创建2行2列取值范围为[0,1)的数组
np.random.rand(2,2)
```
```python
# 创建2行3列，取值范围为标准正态分布的数组
np.random.randn(2,3)
```
#### 生成坐标
**e.g.** 生成二维坐标

```python
row, col = np.indices((2, 3))

row, col
```

```python
print(np.ravel(row))
print(np.ravel(col))
```

**e.g.** 生成二维坐标

```python
x = np.arange(-2, 2, 0.1)
y = np.arange(-2, 2, 0.1)

xv, yv = np.meshgrid(x, y)

xv, yv
```

```python
z = np.sin(xv**2 + yv**2) / (xv**2 + yv**2)
h = plt.contourf(x,y,z)
plt.show()
```

### 读取数据
#### 任意多个位置

```python
a = np.array(range(10))**2

a[[0, 2]]
```

#### Assign all rows/cols of a matrix to variables

```python
a = np.array([[1,2,3],
              [1,2,3]])
```

```python
r1, r2 = a

r1, r2
```

```python
c1, c2, c3 = a.transpose()

c1, c2, c3
```

### 克隆
克隆时候最好把`dtype`也一起带上

```python
"""A = matplotlib.image.imread('pic.jpg')
   注意此时A.dtype为uint8
"""
A = np.zeros((100, 100), dtype=np.uint8)
B = np.zeros_like(A)

"""numpy默认数据类型是float64
   此时A和B的dtype不一样!!!
"""
B = np.zeros(A.shape)
print(A.dtype, "\=", B.dtype)

B1 = np.zeros(A.shape, dtype=A.dtype)
print(A.dtype, "=", B1.dtype)
```





### 运算
#### `np.matmul(A,B)`
请注意W.shape
```python
W = np.arange(2)
X = np.arange(6).reshape(3, 2)
b = 1

res1 = np.matmul(W, X.T) + b
res2 = np.matmul(X, W) + b

print('X.shape:', X.shape, '\nW.shape:', W.shape)
display(Math("W*X^T = \\text{%s}"%(res1)))
display(Math("X*W = \\text{%s}"%(res2)))
```

### 添加或删除


#### Append to an array
```python
a = np.array([1,2,3])

np.append(a, 3)
```

```python
a = np.array([[1,2,3],
              [2,3,4]])
b = np.array([[4],
              [5]])

np.append(a, b, axis=1)
```

#### Stack arrays
- `stack`: Join a sequence of arrays along **a new axis**. (shape会增加一个维度)
- `vstack`: Stack arrays in sequence vertically (row wise).
- `hstack`: Stack arrays in sequence horizontally (column wise).

**e.g.** 1D arrays - `stack`

```python
a = np.array([1, 2, 3])
b = np.array([2, 3, 4])
```

```python
np.stack((a,b), axis=0)
```

```python
np.stack((a,b), axis=1)
```

**e.g.** 1D arrays - `hstack` and `vstack`

```python
np.hstack((a,b))
```

```python
np.vstack((a,b))
```

**e.g.** 2D arrays - `stack`

```python
a = np.array([[1,2,3],
              [2,3,4]])
b = np.array([[1,2,3],
              [2,3,4]])
```

```python
np.stack((a,b), axis=0)
```

```python
np.stack((a,b), axis=1)
```

```python
np.stack((a,b), axis=2)
```

**e.g.** 2D arrays - `vstack` and `hstack`

```python
np.hstack((a,b))
```

#### 删除
```python
a = [1,2,3,4]

a.pop(-1)
a
```

### 获取某些值的位置
#### 可以避免使用for循环
```python
a = np.array([10, 11, 10, 12])
```

```python
a == 10
```

```python
a[(a==10) | (a==12)]
```

**e.g.** 更改array中的某些数值

```python
a[a==11] = 2

a
```

## Misc

### 打印精度设置
参考：
- [A `printf` format reference page (cheat sheet)](https://alvinalexander.com/programming/printf-format-cheat-sheet/)

此外，可以用`np.set_printoptions()`改变全局的设置
```python
print('default:', np.array([2.0]) / 3)

with np.printoptions(precision=2):
    print('precision=2:', np.array([2.0]) / 3)
```

```python
with np.printoptions(formatter={'float':'{:8.2f}'.format}):
    print(np.array([12.2322, 2.2]))
```

```python
a = np.array([1,2,3,1,1])
a[a==1] = 2

a
```

### `np.arctan2()` 弧度、角度计算
`np.arctan()`不可以计算分母为0(90°)的情况，所以最好用`np.arctan2()`
```python
np.arctan2(-10, 0) * 180 / np.pi
```
