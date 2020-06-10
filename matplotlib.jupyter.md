```python
import matplotlib.pyplot as plt
import numpy as np
```

## Magic command
### `%matplotlib inline`
目前版本的jupyter notebook是不需要执行这个命令的（包括`plt.show()`），但是有些情况下，例如执行过`%matplotlib notebook`后，还是需要重新执行`%matplotlib notebook`

```python
plt.plot([1, 1.5, 3.14])
```

<!-- #region -->
### 交互式看图`%matplotlib notebook`
使用这个命令后，可以直接在画出的图上进行“放大”、“缩小”等操作。

由于这个命令会影响自动执行后面cells的非交互式画图，这里就不展示了。但是手动执行jupyter notebook时候不影响（等待当前cell完全绘制完即可）。

```python
# cell 1
%matplotlib notebook

plt.plot([1, 1.6, 3])

#---------------------
# cell 2，恢复非交互式
%matplotlib inline

```
<!-- #endregion -->

## Basic

```python
plt.plot([1, 1.5, 3.14])
```

## Image
### 注意类型和numpy的区别

```python
# 注意此时 A.dtype 为 uint8
A = matplotlib.image.imread('pic.jpg')
B = np.zeros_like(A)

# 此时A和B的dtype不一样
B = np.zeros(A.shape)

# 改成
B = np.zeros(A.shape, dtype=A.dtype)
```