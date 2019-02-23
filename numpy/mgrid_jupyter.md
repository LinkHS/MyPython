`numpy.mgrid` 会生成二维/三维/...坐标，下面代码会生成一组 x-y 坐标（`[0, 0], [0, 1] ... [x-1, y-1]`）:
 ```python
x = 2
y = 3
xy = np.mgrid[0:2, 0:3]
print('shape:', xy.shape)
print(xy)

>>>
shape: (2, 2, 3)
[[[0 0 0]
  [1 1 1]]

 [[0 1 2]
  [0 1 2]]]
 ```

结果形状为 `2*x*y`，会分别得 x 和 y 的矩形坐标值（长宽为 `x*y`）。同样 `np.mgrid[0:x, 0:y, 0:z]` 会生成一组形状为 `3*x*y*z`的 x-y-z 的三维坐标。

```{.python .input  n=1}
import numpy as np
xy_mgrid = np.mgrid[0:3, 0:4]
print('shape:', xy_mgrid.shape)
print(xy_mgrid)
```

`xy_mgrid[0, :, :]` 存储了 x 轴对应的坐标，`xy_mgrid[1, :, :]` 存储了 y 轴对应的坐标

```{.python .input  n=2}
for x, y in zip(xy_mgrid[0, :, :].reshape(-1), xy_mgrid[1, :, :].reshape(-1)):
    print(x, y)
```

我们需要把 xy_mgrid 的第三维转置到第一维（可以参考同目录下的 [transpose_jupyter.md](transpose_jupyter.md)），这样就 x 和 y 坐标就合并了一个个点坐标。

```{.python .input  n=3}
print('T shape:', xy_mgrid.T.shape)
#print(xy_mgrid.T)
print('shape:', xy_mgrid.T.reshape(-1,2).shape)
print(xy_mgrid.T.reshape(-1,2))
```
