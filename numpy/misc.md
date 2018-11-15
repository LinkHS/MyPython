
---
避免使用`for`循环
```
A = np.array([1,2,3,1,1])
A[A==1] = 2
A

> array([2, 2, 3, 2, 2])
```

---
注意 np.array 即使size, shape相同，但是可能dtype不一样
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
