# Pandas

```python
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
```

```python
pd.DataFrame({'red': 4, 'blue': 2}, index=[''])
```

```python
results = np.random.rand(2, 10)

df = pd.DataFrame(results, dtype='float', index=['fair', 'biased'])

print(df)
df
```

```python
df['Ave.'] = df.mean(axis=1)
df
```

# Matplotlib


## 二维直方图
一维直方图统计$X$的数据分布（相同/相近的值放在一组），同理二维直方图统计$(X,Y)$的联合数据分布。注意这里和二维图像不同，二维图像$(x_i,y_i)$是一个坐标，对应这个坐标上的值；而二维直方图的$(x_i, y_i)$对应的二维向量。

例如，我们随机生成一个2行10列的二维数组，第一行代表$X$，第二行代表$Y$：

```python
XY = np.random.randint(0, 2, (2,10))
df = pd.DataFrame(XY, dtype='int', index=['X', 'Y'])

df
```

然后我们统计$XY$值相同的个数：

```python
from collections import Counter

XY_str = ['%d%d'%(x,y) for x,y in XY.transpose()]
XY_cnt = dict(Counter(XY_str))

pd.DataFrame(data=XY_cnt, index=['count'])
```

类似，随机生成两组服从二维高斯分布的数据，中心分别为`(10, 10)`和`(30, 20)`，数量分别为100000和50000个，画出二维直方图：

```python
from numpy.random import multivariate_normal
data = np.vstack([
    multivariate_normal([10, 10], cov=[[3, 2], [2, 3]], size=100000),
    multivariate_normal([30, 20], cov=[[2, 3], [1, 3]], size=50000)
])

plt.title('Linear normalization')
plt.hist2d(data[:, 0], data[:, 1], bins=100)

plt.show()
```

# graphviz
`sudo apt-get install graphviz`

```python
from graphviz import Digraph
# Create Digraph object
dot = Digraph()

# Add nodes
dot.node('1', shape="box")
dot.node('3', shape="circle")
dot.node('2')
dot.node('5')

# Add edges
dot.edges(['12', '13', '35'])

# Visualize the graph
dot
```

```python

```

## 泰勒展开式

$$f(x)=\frac{f\left(x_{0}\right)}{0 !}+\frac{f^{\prime}\left(x_{0}\right)}{1 !}\left(x-x_{0}\right)+\frac{f^{\prime \prime}\left(x_{0}\right)}{2 !}\left(x-x_{0}\right)^{2}$$

$$\begin{aligned}
f(x) &= x^2\\
&= f(x_0)+f^{\prime}(x_0)(x-x_0) + \frac{1}{2} f^{\prime \prime}(x_0) (x-x_0)^{2}\end{aligned}$$

```python
def Square(x):
    return x**2

def TaylorSquare(x, x0):
    return Square(x0) + 2*x0*(x-x0) + (x-x0)**2
```

```python
x = 10

Square(x), TaylorSquare(x, 0), TaylorSquare(x, 100)
```

```python
X = np.linspace(-10, 10)
Y = X**2


dY = 2*X
ddY = np.zeros(len(X)) + 2

plt.plot(X, Y, label='$y=x^2$')
plt.plot(X, dY, label='y\'=$2x$')
plt.plot(X, ddY, label='y\'\'=2')
plt.legend()
plt.show()
```

$$\begin{aligned}
f(x) &= x^2\\
&= f(x_0)+f^{\prime}(x_0)(x-x_0) + \frac{1}{2} f^{\prime \prime}(x_0) (x-x_0)^{2}\end{aligned}$$

```python

```

---
# Temp

```python

```

```python

```

```python

```

```python

```
