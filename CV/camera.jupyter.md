# 相机模型和增强现实
## 针孔相机模型
**相机坐标系**下的一个三维点$\mathbf{X} (X, Y, Z)$映射到二维点$\mathbf{x} (u, v)$的公式如下：

$$\mathbf{x} = \frac{1}{Z}K\mathbf{X}$$

$$
\left(\begin{array}{l}
u \\
v \\
1
\end{array}\right)=\frac{1}{Z}\left(\begin{array}{ccc}
f_{x} & 0 & c_{x} \\
0 & f_{y} & c_{y} \\
0 & 0 & 1
\end{array}\right)\left(\begin{array}{l}
X \\
Y \\
Z
\end{array}\right)
$$

其中$K$为相机内参：

$$
K=\left[\begin{array}{ccc}
f_x & s & c_{x} \\
0 & f_y & c_{y} \\
0 & 0 & 1
\end{array}\right]
$$

这里$f_x = \alpha f_y$，$s$为传感器的偏斜参数。如果成像长宽相等，则$f_x = f_y$，可以忽略$\alpha$。

**世界坐标系**下的一个点$\mathbf{X}_w$则需要加上旋转矩阵$R$和平移向量$t$：

$$
\left[\begin{array}{c}
u \\
v \\
1
\end{array}\right] = \frac{1}{Z}K\left(R \mathbf{X}_{w}+t\right) = \frac{1}{Z} K T \mathbf{X}_{w}
$$

我们设$P = K[R|t]$。

```python
from scipy import linalg
import numpy as np
import pylab
import matplotlib.pyplot as plt
import cv2

class Camera:
    """针孔相机类"""

    def __init__(self, P):
        """初始化P = K[R|t]"""
        self.P = P
        self.K = None
        self.R = None
        self.t = None
        self.c = None

    def project(self, X):
        """
          @X, 4xn的投影点（最后一维为1），并且进行坐标归一化
        """
        x = np.dot(self.P, X)
        for i in range(3):
            x[i] /= x[2]
        return x
```

```python
# 载入点
points = np.loadtxt('data/house.p3d').T
points = np.vstack((points, np.ones(points.shape[1])))

# 设置照相机参数
P = np.hstack((np.eye(3), np.array([[0], [0], [-10]])))
cam = Camera(P)
house_2d = cam.project(points)

# 绘制投影
pylab.figure()
pylab.plot(house_2d[0], house_2d[1], 'k.')
pylab.show()
```

## 相机内参
现实世界的空间点$P=[X, Y, Z]^T$，经过小孔$O$投影之后，落在物理成像平面得到$P'=[X', Y', Z']^T$。设物理成像平面到小孔的距离为$f$(焦距)，那么有：

$$
\frac{Z}{f}=\frac{X}{X^{\prime}}=-\frac{Y}{Y^{\prime}}
$$

因为是倒立的，所有有个负号，我们将成像可以移到另一侧：

$$
\frac{Z}{f}=\frac{X}{X^{\prime}}=\frac{Y}{Y^{\prime}}
$$

于是可以得到：

$$
\begin{array}{l}
X^{\prime}=f \frac{X}{Z} \\
Y^{\prime}=f \frac{Y}{Z}
\end{array}
$$

但是在相机中输出的是图像像素，这需要在成像平面上进行采样和量化，我们引入**像素坐标系**，得到$P'$的像素坐标$[u, v]^T$：

$$
\left\{\begin{array}{l}
u=\alpha X^{\prime}+c_{x} \\
v=\beta Y^{\prime}+c_{y}
\end{array}\right.
$$

可以看到像素坐标系与成像平面之间多了一个缩放和一个平移（相对图像左上方的原点）。设$f_x = \alpha f, f_y = \beta f$：

$$
\left\{\begin{array}{l}
u=f_{x} \frac{X}{Z}+c_{x} \\
v=f_{y} \frac{Y}{Z}+c_{y}
\end{array}\right.
$$

其中，$f$的单位为“米”，$\alpha, \beta$的单位为“像素/米”，所以$f_x, f_y$的单位为“像素”。当图像是正方形时，$f_x = f_y$。

### e.g. 相机内参估算
假设一台相机的焦距$f=4\text{mm}=4\times 10^{-3}\text{m}$，拍摄照片的分辨率为$640\times 480$（像素），传感器尺寸为$ 7400 \text{um} \times 5600 \text{um}$，那么我们可以得到：

$$\alpha = \frac{640}{5856\times 10^{-6}}, \quad \beta = \frac{480}{3276\times 10^{-6}}$$

进一步得到：

$$f_x = \alpha f = 345.95, \quad f_y = \beta f = 342.86$$

现在再看$u, v$的公式，当同样大小的物体越远，即$Z$越大，$u, v$就越小。


## 旋转向量
<span id="jump1">任意旋转</span>都可以用一个旋转轴和一个旋转角来表示，如果一个向量，其方向与旋转轴一致，长度表示旋转角，这样只要一个三维向量即可描述三维空间的旋转。外积（叉积）可以表达两个向量的旋转关系：

![](http://static.zybuluo.com/AustinMxnet/6wyl4jecolb6faysghgzxxp6/image.png)

如图所示，$a, b$的外积$w$可以表示右手系中$a$到$b$的旋转。外积的公式如下：

$$
\boldsymbol{a} \times \boldsymbol{b}=\left[\begin{array}{ccc}
\boldsymbol{i} & \boldsymbol{j} & \boldsymbol{k} \\
a_{1} & a_{2} & a_{3} \\
b_{1} & b_{2} & b_{3}
\end{array}\right]=\left[\begin{array}{c}
a_{2} b_{3}-a_{3} b_{2} \\
a_{3} b_{1}-a_{1} b_{3} \\
a_{1} b_{2}-a_{2} b_{1}
\end{array}\right]=\left[\begin{array}{ccc}
0 & -a_{3} & a_{2} \\
a_{3} & 0 & -a_{1} \\
-a_{2} & a_{1} & 0
\end{array}\right] \boldsymbol{b}
$$

同理$b$到$w$的旋转则表示为$a$，那么对应的旋转矩阵就是：

$$
\left[\begin{array}{ccc}
0 & -a_{3} & a_{2} \\
a_{3} & 0 & -a_{1} \\
-a_{2} & a_{1} & 0
\end{array}\right]
$$

我们可以验证一下$b \times (a \times b)$的结果应该等于$a$：

```python
a = np.array([0, 0, 1])
b = np.array([0, 1, 0])

w = np.cross(a, b)
np.cross(b, w)
```

将旋转轴转换为旋转矩阵的函数如下：

```python
def rotation_matrix(a):
    """创建一个用于围绕向量a轴旋转的三维旋转矩阵"""
    R = np.eye(4)
    R[:3, :3] = linalg.expm([[ 0,     -a[2],  a[1]], 
                             [ a[2],   0,    -a[0]],
                             [-a[1],   a[0],  0]])
    return R
```

### e.g. 旋转
我们随机生成一个旋转向量，在三维空间中对house进行旋转，再投影到二维图像上（红色为旋转之前的图像）：

```python
pylab.figure()
pylab.plot(house_2d[0], house_2d[1], 'r.')

r = np.array([0.12, 0.62, 0.83])
rot = rotation_matrix(r)
cam.P = np.dot(P, rot)
x = cam.project(points)

pylab.plot(x[0], x[1], 'k.')
pylab.show()
```

我们可以将一次旋转分为多次（这里用20次）来展示其变化过程：

```python
r = 0.05 * r
rot = rotation_matrix(r)
cam.P = cam_P

pylab.figure()
for t in range(20):
    cam.P = np.dot(cam.P, rot)
    x = cam.project(points)
    pylab.plot(x[0], x[1], 'k.')
pylab.show()
```

## 相机矩阵分解

我们知道已知$K, R, t$很容易求得矩阵$P = K[R|t]$，但是如果知道$P$如何恢复内参$K$以及相机的位置$t$和姿势$R$呢？我们将使用一种矩阵因子分解的方法，称为RQ因子分解。

RQ因子分解的结果并不是唯一的。在该因子分解中，分解的结果存在符号二义性。由于我们需要限制旋转矩阵R为正定的（否则，旋转坐标轴即可），具体可参考[Dissecting the Camera Matrix](http://ksimek.github.io/2012/08/14/decompose/)。

```python
def factor(P):
    """将相机矩阵P分解为K，R，t
      @P, 4x4矩阵
    """
    
    K, R = linalg.rq(P[:, :3])
    
    # Make sure K has a positive determinant.
    T = np.diag(np.sign(np.diag(K)))
    if linalg.det(T) < 0:
        T[1, 1] *= -1
    
    K = np.dot(K, T)
    R = np.dot(T, R) # T的逆矩阵为其自身
    t = np.dot(linalg.inv(K), P[:, 3])
    
    return K, R, t
```

下面测试一下`factor()`，分解输出的`K_`、`R`、`t`应该等于`K`、`Rt`：

```python
K = np.array([[1000, 0, 500],
              [0, 1000, 300],
              [0, 0, 1]])
tmp = rotation_matrix([0, 0, 1])[:3, :3]
Rt = np.hstack((tmp, np.array([[50], [40], [30]])))
print('Rt:\n', Rt.round(2))

K_, R, t = factor(np.dot(K, Rt))
print('K_:\n', K_.round(2))
print('R:\n', R.round(2))
print('t:\n', t.round(2))
```

## 计算相机中心
给定相机投影矩阵$P$，我们可以计算出空间上
相机的中心$C$是一个三维点，满足约束$PC=0$。对于

```python

```

# 三维空间刚体运动
刚体运动保证了同一个向量在各个坐标系下的长度和夹角都不会发生变化，这种变换称为**欧氏变换**。欧氏变换由一个旋转和一个平移两部分组成。

## 旋转矩阵
设某个单位正交基（3维坐标系）$(e_1, e_2, e_3)$经过一次旋转，变成了$e'_1, e'_2, e'_3$。假设一个向量$a$在两个坐标系下的坐标分别为$[a_1, a_2, a_3]^T$和$[a'_1, a'_2, a'_3]^T$，注意这个向量并没有随着坐标系一起旋转，否则坐标不会变化。根据坐标系的定义，有：

$$
\left[ {e}_{1},  {e}_{2},  {e}_{3}\right]\left[\begin{array}{c}
a_{1} \\
a_{2} \\
a_{3}
\end{array}\right]=\left[ {e}_{1}^{\prime},  {e}_{2}^{\prime},  {e}_{3}^{\prime}\right]\left[\begin{array}{c}
a_{1}^{\prime} \\
a_{2}^{\prime} \\
a_{3}^{\prime}
\end{array}\right]
$$

对上式同时左乘$[e_1^T, e_2^T, e_2^T]$，因为$e$是单位向量，所以左边的系数变成了单位矩阵：

$$
\left[\begin{array}{l}
a_{1} \\
a_{2} \\
a_{3}
\end{array}\right]=\left[\begin{array}{lll}
e_{1}^{T} e_{1}^{\prime} & e_{1}^{T} e_{2}^{\prime} & e_{1}^{T} e_{3}^{\prime} \\
e_{2}^{T} e_{1}^{\prime} & e_{2}^{T} e_{2}^{\prime} & e_{2}^{T} e_{3}^{\prime} \\
e_{3}^{T} e_{1}^{\prime} & e_{3}^{T} e_{2}^{\prime} & e_{3}^{T} e_{3}^{\prime}
\end{array}\right]\left[\begin{array}{c}
a_{1}^{\prime} \\
a_{2}^{\prime} \\
a_{3}^{\prime}
\end{array}\right] \triangleq R a^{\prime}
$$

矩阵$R$由两组基之间的内积组成，**描述了旋转前后同一个向量的坐标变换关系**。我们一开始是让坐标系旋转（向量$a$不旋转），现在是让同样的旋转作用在$a$上（坐标系不旋转），这两个旋转是一样的，都是$R$。旋转矩阵的集合定义：

$$
S O(n)=\left\{ {R} \in \mathbb{R}^{n \times n} \mid  {R}  {R}^{T}= {I}, \operatorname{det}( {R})=1\right\}
$$

旋转矩阵的特殊性质：
1. 行列式为1（$\text{det}(R) = 1$），因为是刚体运动，不能改变长度和夹角。
2. 正交矩阵（$R^{-1} = R^{T}$），因为反向旋转恢复原始姿态（$R^{-1}R=I$），加上正交基关系$R^TR=I$。


## 变换矩阵
欧氏变换除了旋转，还有平移。把旋转和平移合到一起得到完整的一次欧氏变换：

$$a' = Ra + t$$

虽然公式很简洁，但是有个问题，如果进行了两次变换：

$$
b=R_{1} {a}+{t}_{1}, \quad {c}= {R}_{2} b+ {t}_{2}
$$

如果想描述为一次$a$到$c$则变为了：

$$c = R_2(R_1a+t_1)+t_2$$

更多次之后的合并形式会过于复杂，所以引入齐次坐标：

$$
\left[\begin{array}{l}
a^{\prime} \\
1
\end{array}\right]=\left[\begin{array}{ll}
R & t \\
0^{T} & 1
\end{array}\right]\left[\begin{array}{l}
a \\
1
\end{array}\right] \triangleq T\left[\begin{array}{l}
a \\
1
\end{array}\right]
$$

矩阵$T$称为变换矩阵（Transform Matrix）。这样两次变换就可以表示为：

$$
\tilde{ {b}}= {T}_{1} \tilde{ {a}},\ \tilde{ {c}}= {T}_{2} \tilde{ {b}} \quad \Rightarrow \tilde{ {c}}= {T}_{2}  {T}_{1} \tilde{ {a}}
$$

同样变换矩阵$T$的合集：

$$
S E(3)=\left\{{T}=\left[\begin{array}{cc}
{R} & {t} \\
{0}^{T} & 1
\end{array}\right] \in \mathbb{R}^{4 \times 4} \mid {R} \in S O(3), {t} \in \mathbb{R}^{3}\right\}
$$

与$SO(3)$一样，**该矩阵的逆表示一个反向的变换**：

$$
{T}^{-1}=\left[\begin{array}{cc}
{R}^{T} & -{R}^{T} {t} \\
{0}^{T} & 1
\end{array}\right]
$$


## 旋转向量
[上文](#jump1) 我们介绍了如何用外积表达两个向量的旋转关系。对于坐标系的旋转，任意旋转都可以用**一个旋转轴和一个旋转角**来刻画。这种表示法只需一个三维向量即可描述三维空间的旋转，**其方向与旋转轴一致，而长度等于旋转角**。同样，对于变换矩阵，我们使用一个旋转向量和一个平移向量即可表达一次变换。**这时的维数正好是六维，正好等于三维刚体运动的自由度（6），相比之下$SE(3)$需要十六个量**。

### 旋转向量与旋转矩阵

旋转向量$\theta n$的旋转轴为$n$，角度为$\theta$，对应的旋转矩阵$R$等于：

$$
R = \cos \theta I + (1-\cos \theta)nn^T + \sin \theta n^{\wedge}
$$

反之，旋转矩阵$R$转为旋转向量$\theta n$，先求$\theta$：

$$
\begin{aligned}
\operatorname{tr}({R}) &=\cos \theta \operatorname{tr}({I})+(1-\cos \theta) \operatorname{tr}\left({n} {n}^{T}\right)+\sin \theta \operatorname{tr}\left({n}^{\wedge}\right) \\
&=3 \cos \theta+(1-\cos \theta) \\
&=1+2 \cos \theta
\end{aligned}
$$

得到：

$$
\theta=\arccos \left(\frac{\operatorname{tr}(R)-1}{2}\right)
$$

然后求旋转轴$n$，由于**旋转轴经过旋转之后不变**，即$Rn=n$，可知转轴$n$是矩阵$R$特征值为1对应的特征向量。求解方程$Rn=n$，再归一化就得到了旋转轴。


## 欧拉角

欧拉角提供了一种非常直观的方式来描述旋转：它使用了三个分离的转角，把一个旋转分解成三次绕不同轴的旋转。这里假设**按照XYZ的顺序进行旋转**，当围绕X轴旋转$\psi$的旋转矩阵如下：

$$
R_{x}(\psi)=\left[\begin{array}{ccc}
1 & 0 & 0 \\
0 & \cos \psi & -\sin \psi \\
0 & \sin \psi & \cos \psi
\end{array}\right]
$$

向量$a=[x, y, z]^T$围绕X轴旋转$\psi$，相当于在YZ平面上旋转变化，只会影响Y和Z轴的坐标，不会改变X轴坐标：

$$
a' =
R_{x}(\psi)\ a =\left[\begin{array}{ccc}
1 & 0 & 0 \\
0 & \cos \psi & -\sin \psi \\
0 & \sin \psi & \cos \psi
\end{array}\right]
\left[\begin{array}{c}
x \\
y \\
z
\end{array}\right]
=
\left[\begin{array}{c}
x \\
y\cos \psi - z \sin \psi \\
y\sin \psi + z \cos \psi
\end{array}\right]a
$$

同样可以得到围绕Y轴和Z轴旋转的矩阵：

$$
R_{y}(\theta)=\left[\begin{array}{ccc}
\cos \theta & 0 & \sin \theta \\
0 & 1 & 0 \\
-\sin \theta & 0 & \cos \theta
\end{array}\right]
$$

$$
R_{z}(\phi)=\left[\begin{array}{ccc}
\cos \phi & -\sin \phi & 0 \\
\sin \phi & \cos \phi & 0 \\
0 & 0 & 1
\end{array}\right]
$$

将三个轴的旋转**按照XYZ顺序合起来**：
$$
R =R_{z}(\phi) R_{y}(\theta) R_{x}(\psi)
=\left[\begin{array}{ccc}
\cos \theta \cos \phi & \sin \psi \sin \theta \cos \phi-\cos \psi \sin \phi & \cos \psi \sin \theta \cos \phi+\sin \psi \sin \phi \\
\cos \theta \sin \phi & \sin \psi \sin \theta \sin \phi+\cos \psi \cos \phi & \cos \psi \sin \theta \sin \phi-\sin \psi \cos \phi \\
-\sin \theta & \sin \psi \cos \theta & \cos \psi \cos \theta
\end{array}\right]
$$

### 欧拉角与旋转矩阵
由上面的公式就可以直接求得旋转矩阵：

$$
R=\left[\begin{array}{lll}
R_{11} & R_{12} & R_{13} \\
R_{21} & R_{22} & R_{23} \\
R_{31} & R_{32} & R_{33}
\end{array}\right]
$$

<!-- #region -->
### 万向锁
网上分析万向锁的文章很多，但是总觉的用机械旋转图来解释并不对。这里从另一个角度看下。一个向量$a=[x, y, z]^T$，上文说了**围绕X轴旋转只会改变Y轴和Z轴的坐标**：

$$
a'= \left[\begin{array}{c}
x \\
y\cos \psi - z \sin \psi \\
y\sin \psi + z \cos \psi
\end{array}\right]\ a 
= \left[\begin{array}{c}
x \\
y'\\
z'
\end{array}\right]$$

第二次**围绕Y轴只会改变X轴和Z轴的坐标**：

$$
\begin{bmatrix}
 \cos \theta & 0 & \sin \theta  \\
 0 & 1 & 0 \\
 -\sin \theta & 0 & \cos \theta 
\end{bmatrix}
\begin{bmatrix}
 x \\
 y' \\
 z'
\end{bmatrix} =
\begin{bmatrix}
 x \cos \theta + z' \sin \theta \\
 y' \\
 - x \sin \theta + z' \cos \theta 
\end{bmatrix}
= 
\begin{bmatrix}
 x' \\
 y' \\
 z''
\end{bmatrix}
$$

同理第三次**围绕Z轴只会改变X轴和Y轴的坐标**，结果等于：
$$\begin{bmatrix}
 x'' \\
 y'' \\
 z''
\end{bmatrix}
$$


从结果可以看出，正常情况下$x, y, z$的坐标被改变了两次。**问题来了，假设第二次围绕Y轴旋转了90度**，得到：

$$
R_{y}(\theta) \ a' =\left[\begin{array}{ccc}
0  & 0 & 1 \\
0  & 1 & 0 \\
-1 & 0 & 0
\end{array}\right]\ 
\left[\begin{array}{c}
x \\
y'\\
z'
\end{array}\right]
= \left[\begin{array}{c}
z' \\
y' \\
-x
\end{array}\right]
$$

第二次围绕Y轴转换的结果$z'$等于原始的$-x$。继续第三次围绕Z轴旋转，这次只会改变X轴和Y轴的坐标得到：

$$
\left[\begin{array}{c}
x'' \\
y'' \\
-x
\end{array}\right]
$$

注意$x''$和$y''$都是由$z'$和$y'$线性组合得到（系数为$\sin \phi$和$\cos \phi$），而$z'$和$y'$都是由$y$和$z$线性组着得到，这意味着第三次旋转围绕的Z轴等于第一次旋转时候围绕的X轴，仅仅需要两次选择旋转即可：第一次围绕X轴旋转的角度等于$\psi \pm \phi$，第二次围绕Y轴旋转的角度等于90。三次旋转变成了二次旋转，丢失了一个自由度Z轴！
<!-- #endregion -->

## TODO
https://docs.opencv.org/master/d9/dab/tutorial_homography.html#projective_transformations


# 图像到图像的映射
## 单应性变换
单应性变换是将一个平面（图像）内的点映射到另一个平面（图像）内的二维投影变换，记作$\mathbf{x}'=H\mathbf{x}$：

$$
\left[\begin{array}{l}
x^{\prime} \\
y^{\prime} \\
w^{\prime}
\end{array}\right]=\left[\begin{array}{lll}
h_{1} & h_{2} & h_{3} \\
h_{4} & h_{5} & h_{6} \\
h_{7} & h_{8} & h_{9}
\end{array}\right]\left[\begin{array}{l}
x \\
y \\
w
\end{array}\right]
$$

由于二维图像丢失了深度信息，所以$\mathbf{x}=[x, y, w]=[ax, ay, aw]=[x/w, y/w, 1]$表示同一个二维点，所以单应性矩阵$H$仅仅依赖于尺度（不含深度），共有8个独立的自由度。

```python
def normalize(points):
    """在齐次坐标下，对n组二维点进行归一化，使最后一行为1
      @points, 3 x n
    """
    for row in points:
        row /= points[-1]
    return points

def make_homog(points):
    """将n组二维点转换为齐次坐标
      @points, 2 x n
    """
    return np.vstack((points, np.ones((1, points.shape[1]))))
```

```python
xs = np.array([[1, 2], 
               [2, 3],
               [3, 3]], dtype=np.float)

normalize(xs)
```

```python
for x in xs:
    print(x/2)
```

## 直接线性变换（DLT）
单应性矩阵$H$可以由两幅图像（或者平面）中对应“点对”计算出来。$H$有8个自由度，每组对应点对可以写出两个方程，分别对应于$x$和$y$坐标。因此，计算$H$需要４组对应点对：

$$
\left[\begin{array}{ccccccccc}
-x_{1} & -y_{1} & -1 & 0 & 0 & 0 & x_{1} x_{1}^{\prime} & y_{1} x_{1}^{\prime} & x_{1}^{\prime} \\
0 & 0 & 0 & -x_{1} & -y_{1} & -1 & x_{1} y_{1}^{\prime} & y_{1} y_{1}^{\prime} & y_{1}^{\prime} \\
-x_{2} & -y_{2} & -1 & 0 & 0 & 0 & x_{2} x_{2}^{\prime} & y_{2} x_{2}^{\prime} & x_{2}^{\prime} \\
0 & 0 & 0 & -x_{2} & -y_{2} & -1 & x_{2} y_{2}^{\prime} & y_{2} y_{2}^{\prime} & y_{2}^{\prime} \\
& \vdots & & \vdots & & \vdots & & \vdots &
\end{array}\right]\left[\begin{array}{l}
h_{1} \\
h_{2} \\
h_{3} \\
h_{4} \\
\vdots
\end{array}\right]=\mathbf{0}
$$

即$Ah=0$，我们可以使用SVD找到$H$的最小二乘解。因为算法的稳定性取决于坐标的表示情况和数值计算的问题，所以这里对所有输入的点进行归一化，使其均值为0，方差为1：

```python
def H_from_points(fp, tp):
    '''Find H such that H * fp = tp.
    H has eight degrees of freedom, so this needs at least 4 points in fp and tp.
    @fp, tp: 3xn, 最后一维为1
    '''
    if fp.shape != tp.shape:
        raise RuntimeError('number of points do not match')

    # condition（归一化）:
    # -from
    m = np.mean(fp[:2], axis=1)
    maxstd = np.max(np.std(fp[:2], axis=1)) + 1e-9
    C1 = np.diag([1/maxstd, 1/maxstd, 1])
    C1[0, 2] = -m[0] / maxstd
    C1[1, 2] = -m[1] / maxstd
    fp = np.dot(C1, fp)

    # -to
    m = np.mean(tp[:2], axis=1)
    maxstd = np.max(np.std(tp[:2], axis=1)) + 1e-9
    C2 = np.diag([1/maxstd, 1/maxstd, 1])
    C2[0, 2] = -m[0] / maxstd
    C2[1, 2] = -m[1] / maxstd
    tp = np.dot(C2, tp)

    correspondences_count = fp.shape[1]
    A = np.zeros((2 * correspondences_count, 9))
    for i in range(correspondences_count):
        A[2 * i] = [-fp[0][i], -fp[1][i], -1, 0, 0, 0,
                    tp[0][i] * fp[0][i], tp[0][i] * fp[1][i], tp[0][i]]
        A[2 * i + 1] = [0, 0, 0, -fp[0][i], -fp[1][i], -1,
                        tp[1][i] * fp[0][i], tp[1][i] * fp[1][i], tp[1][i]]

    U, S, V = np.linalg.svd(A)
    H = V[8].reshape((3, 3))

    # decondition（反归一化）
    H = np.dot(np.linalg.inv(C2), np.dot(H, C1))

    # 尺度归一化
    return H / H[2, 2]
```

### e.g. 图像对齐
有一本书因为拍摄视角原因在图中呈现出“倾斜”姿态，现在我们可以通过透视变换（Perspective Correction using Homography）来对齐这本书。

```python
im_src = cv2.imread('data/book_perspective.JPG')
plt.imshow(im_src[:,:,::-1])
plt.show()
```

我们可以找出这本书4个顶点在原图中的坐标，然后映射到500x600大小的图中（这里假设长宽比为5:6，如果知道书本真实的长宽比最好）。这里我们先用OpenCV的`findHomography()`测试一下：

```python
pts_src = np.array([[486, 79], [854, 219], [190, 461], [699, 700]]) # col, row
pts_dst = np.array([[0, 0],    [500, 0],   [0, 600],   [500, 600]])

h, status = cv2.findHomography(pts_src, pts_dst)
im_out = cv2.warpPerspective(im_dst, h, (500, 600))

plt.imshow(im_out[:,:,::-1])
plt.show()
```

注意由于原图中的封面有所弯曲，所以映射的图像也有点弯曲。比较惊喜的是原图中的"Understanding"已经看不出来，但是被“校正”后还是很清楚的。下面我们用自己写的`H_from_points()`测试求得的$h$是否一致：

```python
pts_src1 = make_homog(pts_src.T)
pts_dst1 = make_homog(pts_dst.T)

(h - H_from_points(pts_src1, pts_dst1)).round(2)
```

## 仿射变换
如下所示，仿射变换具有6个自由度，变形能力稍微弱些，我们需要三个对应点对来估计矩阵$H$。虽然仿射变换可以用上面的DLT算法估计得出，但是这里用另一种方法（参见"Multiple View Geometry in Computer Vision"第130页）。

$$
\left[\begin{array}{l}
x^{\prime} \\
y^{\prime} \\
1
\end{array}\right]=\left[\begin{array}{lll}
a_{1} & a_{2} & t_{x} \\
a_{3} & a_{4} & t_{y} \\
0 & 0 & 1
\end{array}\right]\left[\begin{array}{l}
x \\
y \\
1
\end{array}\right] \quad \text { 或 } \quad x^{\prime}=\left[\begin{array}{ll}
\boldsymbol{A} & \boldsymbol{t} \\
\mathbf{0} & 1
\end{array}\right] \boldsymbol{x}
$$

```python
def Haffine_from_points(fp, tp):
    '''Find affine H such that H * fp = tp.
    H has six degrees of freedom, so this needs at least 3 points in fp and tp.
    @fp, @tp: 3xn, 最后一维为1
    '''
    if fp.shape != tp.shape:
        raise RuntimeError('number of points do not match')

    # condition:
    # -from
    m = numpy.mean(fp[:2], axis=1)
    maxstd = numpy.max(numpy.std(fp[:2], axis=1)) + 1e-9
    C1 = numpy.diag([1/maxstd, 1/maxstd, 1])
    C1[0, 2] = -m[0] / maxstd
    C1[1, 2] = -m[1] / maxstd
    fp_cond = numpy.dot(C1, fp)

    # -to
    m = numpy.mean(tp[:2], axis=1)
    maxstd = numpy.max(numpy.std(tp[:2], axis=1)) + 1e-9
    C2 = numpy.diag([1/maxstd, 1/maxstd, 1])
    C2[0, 2] = -m[0] / maxstd
    C2[1, 2] = -m[1] / maxstd
    tp_cond = numpy.dot(C2, tp)

    A = numpy.concatenate((fp_cond[:2], tp_cond[:2]), axis=0)
    U, S, V = numpy.linalg.svd(A.T)

    tmp = V[:2].T
    B = tmp[:2]
    C = tmp[2:4]

    tmp2 = numpy.concatenate((numpy.dot(C, numpy.linalg.pinv(B)),
                              numpy.zeros((2, 1))),
                             axis=1)
    H = numpy.vstack((tmp2, [0, 0, 1]))

    # decondition
    H = numpy.dot(numpy.linalg.inv(C2), numpy.dot(H, C1))
    return H / H[2, 2]
```

### e.g. 仿射变换

```python
cols, rows = 250, 250
img = np.full((cols, rows, 3), 255, np.uint8)
img = cv2.rectangle(img, (50, 50), (200, 200), (0, 255, 0), 2)

pts1 = np.float32([[50, 50], 
                   [200, 50],  
                   [50, 200]]) 
  
pts2 = np.float32([[10, 100], 
                   [200, 50],  
                   [100, 250]])

# 仿射变换
M = cv2.getAffineTransform(pts1[0:3], pts2[0:3]) 
im_out = cv2.warpAffine(img, M, (cols, rows)) 

# 画图
fig, axs = plt.subplots(1, 2, figsize=(9, 5))
axs[0].imshow(img[:,:,::-1])
axs[1].imshow(im_out[:,:,::-1])
plt.show()
```

从结果可以看出，正方形的三个顶点（左上、右上、右下）`pts1`被映射到了我们设置的位置`pts2`，我们可以将`M`打印出来，并且和`pts1`一起带入$\mathbf{x'}=M\mathbf{x}$验证一下结果是否等于`pts2`：

```python
print('M:\n', M)

np.dot(M, np.vstack([pts1.T, [1, 1, 1]]))
```

### e.g. 仿射变换 VS 单应性变换
仿射变换是单应性变换的一种情况，但是只有6个自由度，而单应性变换有9个自由度（但是2维图像会丢失一个自由度）：

$$
\left[\begin{array}{l}
x^{\prime} \\
y^{\prime} \\
1
\end{array}\right]=\left[\begin{array}{lll}
a_{1} & a_{2} & t_{x} \\
a_{3} & a_{4} & t_{y} \\
0 & 0 & 1
\end{array}\right]\left[\begin{array}{l}
x \\
y \\
1
\end{array}\right]
\quad
\text{VS}
\quad
\left[\begin{array}{l}
x^{\prime} \\
y^{\prime} \\
w^{\prime}
\end{array}\right]=\left[\begin{array}{lll}
h_{1} & h_{2} & h_{3} \\
h_{4} & h_{5} & h_{6} \\
h_{7} & h_{8} & h_{9}
\end{array}\right]\left[\begin{array}{l}
x \\
y \\
w
\end{array}\right]
$$

我们可以把上个例子（e.g. 仿射变换）中的`M`再加上三个自由度后看下相比原来的`M`会有怎样的变换效果：

```python
H1 = np.vstack((M, [0.004, 0, 1]))
H2 = np.vstack((M, [0.0,   0, 2]))
H3 = np.vstack((M, [0.004, 0, 2]))

# 仿射变换
im_out1 = cv2.warpPerspective(img, H1, (cols, rows)) 
im_out2 = cv2.warpPerspective(img, H2, (cols, rows)) 
im_out3 = cv2.warpPerspective(img, H3, (cols, rows)) 

# 画图
fig, axs = plt.subplots(1, 4, figsize=(14, 5))
axs[0].imshow(im_out[:,:,::-1])
axs[1].imshow(im_out1[:,:,::-1])
axs[2].imshow(im_out2[:,:,::-1])
axs[3].imshow(im_out3[:,:,::-1])

plt.show()

with np.printoptions(precision=3, suppress=True):
    print('M:\n',  M)
    print('\nH1:\n', H1)
    print('\nH2:\n', H2)
    print('\nH3:\n', H3)
```

我们可以看下原来三个顶点转换后的坐标，注意每一列是一个图像坐标$(u,v,w)$，所以要除以第三行归一化得到$(x, y, 1)=(u/w, v/w, 1)$：

```python
pts1_T = np.vstack([pts1.T, [1, 1, 1]])
print(np.dot(M, pts1_T), '\n')
print(np.dot(H1, pts1_T), '\n')
print(np.dot(H2, pts1_T), '\n')
print(np.dot(H3, pts1_T), '\n')
```

```python
img = cv2.imread('data/book_frontal.JPG') 
rows, cols, ch = img.shape 

pts1 = np.float32([[486, 79],  [854, 219], [190, 461], [699, 700]]) # col, row
pts2 = np.float32([[0, 0],    [500, 0],   [0, 600],   [500, 600]])

M = cv2.getAffineTransform(pts1[0:3], pts2[0:3]) 
im_dst = cv2.warpAffine(img, M, (cols, rows)) 
plt.imshow(im_dst[:,:,::-1])
plt.show()
M
```

```python
h, status = cv2.findHomography(pts1, pts2)
im_out = cv2.warpPerspective(img, h, (cols, rows))

plt.imshow(im_out[:,:,::-1])
plt.show()
h
```

```python
im_src = cv2.imread('data/book_frontal.JPG')
im_dst = cv2.imread('data/book_perspective.JPG')

fig, axs = plt.subplots(1, 2, figsize=(14, 9))
axs[0].imshow(im_src[:,:,::-1])
axs[1].imshow(im_dst[:,:,::-1])
plt.show()
```

```python
pts_src1 = make_homog(pts_src.T)
pts_dst1 = make_homog(pts_dst.T)

h, H_from_points(pts_src1, pts_dst1)
```

```python

# tl, tr, bl, br
(76, 691) 
(60, 50)
(917, 691)
(929, 39)

(486, 79)
(853, 218)
(190, 461)
(699, 700)
```

```python

```

```python

```

```python

```

```python

```
