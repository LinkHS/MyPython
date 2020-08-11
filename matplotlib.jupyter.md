# Matplotlib

![](http://static.zybuluo.com/AustinMxnet/tqi0tspg5phwlaakhptpwg8w/image.png)

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

### 调整图像之间的间隔

```python
fig, axs = plt.subplots(1, 2)

axs[0].plot(range(10))

plt.subplots_adjust(wspace=0.5)
```

## Image
### 注意类型和numpy的区别

```python
import matplotlib

# 注意此时 A.dtype 为 uint8
A = matplotlib.image.imread('_files/liuyifei.jpg')
B = np.zeros_like(A)

# 此时A和B的dtype不一样!!!
B = np.zeros(A.shape)
print(A.dtype, B.dtype)

# 改成
B = np.zeros(A.shape, dtype=A.dtype)
print(A.dtype, B.dtype)
```
