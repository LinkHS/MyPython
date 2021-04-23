```python
import matplotlib.pyplot as plt
import numpy as np
import geatpy as ea
import time

from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from pprint import pprint
```

# Geatpy
Geatpy的进化算法框架由四个大类构成：算法模板类(Algorithm)、种群类(Population)、多染色体混合编码种群类(PsyPopulation)以及问题类 (Problem)。其中Population类和PsyPopulation类是可以直接被实例化成对象去来使用的类；Algorithm 类和 Problem 类是父类，需要实例化其子类来使用。

![image](http://static.zybuluo.com/AustinMxnet/tv1fmmi1nuqrh1bqgu0jwcd6/image.png)

`Population`类拥有`Chrom`、`Phen`、`ObjV`、`CV`、`FitnV`等重要属性，分别指代种群的染色体矩阵、染色体解码后得到的表现型矩阵、目标函数值矩阵、违反约束程度矩阵、适应度列向量（详见`Population.py`关于种群类的定义）。

为了解释Geatpy的细节，第一个问题使用脚本编程法，后面的问题使用**面向对象**编程。


## 脚本编程法
标准测试函数---McCormick 函数的最小化搜索问题为例，其表达式为：

$f(x,y) = sin(x+y) + (x-y)^2 -1.5x + 2.5y + 1 $ 

它是一个二元函数，它具有一个全局极小点：$f(-0.54719, -1.54719)=-1.9133$，函数图像如下：

```python
X = np.arange(-5, 5, 0.25)
Y = np.arange(-5, 5, 0.25)
X, Y = np.meshgrid(X, Y)
Z = np.sin(X+Y) + np.square(X-Y) - 1.5*X + 2.5*Y + 1

fig = plt.figure()
ax = Axes3D(fig)
ax.plot_surface(X, Y, Z, cmap=cm.viridis)

plt.show()
```

拟采用带精英保留的遗传算法“Elitist Reservation GA”来求解。

### 定义目标函数

```python
def aim(Phen):
    """传入种群染色体矩阵解码后的基因表现型矩阵
    """
    x1 = Phen[:, [0]] # 取出第一列， 得到所有个体的第一个自变量
    x2 = Phen[:, [1]] # 取出第二列， 得到所有个体的第二个自变量
    return np.sin(x1 + x2) + (x1 - x2) ** 2 - 1.5 * x1 + 2.5 * x2 +1
```

### 定义变量x&y的范围

```python
x1 = [-1.5, 4] # 第一个决策变量范围
x2 = [-3, 4]   # 第二个决策变量范围
b1 = [1, 1]    # 第一个决策变量边界，1包含边界，0不包含
b2 = [1, 1]    # 第二个决策变量边界，1包含边界，0不包含

ranges = np.vstack([x1, x2]).T   # 自变量的范围矩阵（第一行为下界， 第二行为上界）
borders = np.vstack([b1, b2]).T  # 自变量的边界矩阵
varTypes = np.array([0, 0])      # 决策变量的类型，0连续，1离散

pprint(ranges)
```

### 染色体编码设置

```python
Encoding = 'BG'     # 'BG'表示采用二进制/格雷编码
codes = [1, 1]      # 编码方式，两个1:变量均使用格雷编码
precisions = [6, 6] # 编码精度，小数点后6位
scales = [0, 0]     # 0:算术刻度，1:采用对数刻度

# 创建译码矩阵
FieldD = ea.crtfld(Encoding, varTypes, ranges,
                   borders, precisions, codes, scales)

pprint(FieldD)
```

### 遗传算法参数设置

```python
NIND = 20     # 种群个体数目
MAXGEN = 100  # 最大遗传代数
maxormins = np.array([1])  # 1:目标函数最小化，-1:最大化
selectStyle = 'sus'  # 随机抽样选择
recStyle = 'xovdp'   # 两点交叉
mutStyle = 'mutbin'  # 采用二进制染色体的变异算子
Lind = int(np.sum(FieldD[0, :]))  # 计算染色体长度
pc = 0.9       # 交叉概率
pm = 1 / Lind  # 变异概率
obj_trace = np.zeros((MAXGEN, 2))    # 目标函数值记录器
var_trace = np.zeros((MAXGEN, Lind)) # 染色体记录器，历代最优个体的染色体
```

### 遗传算法进化

```python
start_time = time.time()

Chrom = ea.crtpc(Encoding, NIND, FieldD)  # 生成种群染色体矩阵
variable = ea.bs2real(Chrom, FieldD)      # 对初始种群进行解码
ObjV = aim(variable)        # 计算初始种群个体的目标函数值
best_ind = np.argmin(ObjV)  # 计算当代最优个体的序号

for gen in range(MAXGEN):
    FitnV = ea.ranking(maxormins * ObjV)  # 根据目标函数大小分配适应度值
    SelCh = Chrom[ea.selecting(selectStyle, FitnV, NIND-1), :]  # 选择
    SelCh = ea.recombin(recStyle, SelCh, pc)  # 重组
    SelCh = ea.mutate(mutStyle, Encoding, SelCh, pm)  # 变异

    # 把父代精英个体与子代的染色体进行合并，得到新一代种群
    Chrom = np.vstack([Chrom[best_ind, :], SelCh])
    Phen = ea.bs2real(Chrom, FieldD)  # 对种群进行解码
    ObjV = aim(Phen)

    """记录"""
    best_ind = np.argmin(ObjV)  # 当代最优个体的序号
    obj_trace[gen, 0] = np.sum(ObjV) / ObjV.shape[0]  # 当代种群的目标函数均值
    obj_trace[gen, 1] = ObjV[best_ind]  # 当代种群最优个体目标函数值
    var_trace[gen, :] = Chrom[best_ind, :]  # 记录当代种群最优个体的染色体

end_time = time.time()  # 结束计时
ea.trcplot(obj_trace, [['average value of the whole population',
                        'value of the best individual']])
```

```python
best_gen = np.argmin(obj_trace[:, [1]])
variable = ea.bs2real(var_trace[[best_gen], :], FieldD)

print('最优解的目标函数值：', obj_trace[best_gen, 1])
print('最优解的决策变量值为：')
for i in range(variable.shape[1]):
    print('x', str(i), '=', variable[0, i])
print('用时：', end_time - start_time, '秒')
```

## 带约束的单目标优化
待优化的问题模型如下：

$$\max f\left(x_{1}, x_{2}, x_{3}\right)=4 x_{1}+2 x_{2}+x_{3}$$

$$\begin{aligned}
\text{s.t.}\ &2 x_{1}+x_{2} \leq 1 \\
&x_{1}+2 x_{3} \leq 2 \\
&x_{1}+x_{2}+x_{3}=1 \\
&x_{1} \in[0,1], x_{2} \in[0,1], x_{3} \in(0,2)
\end{aligned}
$$

这是一个带不等式约束和等式约束的单目标最大化优化问题，存在多个局部最优解，对进化算法具有一定的挑战性。全局最优解为$f(0.5, 0, 0.5) = 2.5$。这里拟采用差分进化（Differential Evolution）算法`DE/best/1/L`来求解该问题 (算法模板源码详见“`soea_DE_best_1_L_templet.py`”)，此时只需要进行编写问题子类和编写执行脚本两个步骤即可完成问题的求解。

### 问题建模
这里我们使用面向对象编程，通过继承`Problem`类完成对问题模型的描述：

```python
import numpy as np
import geatpy as ea

class MyProblem(ea.Problem): 
    def __init__(self):
        name = 'MyProblem'
        M = 1             # 优化目标维数
        maxormins = [-1]  # 目标最小/最大化，1:min；-1:max
        Dim = 3           # 决策变量维数
        varTypes = [0] * Dim  # 决策变量类型，0:连续；1:离散
        lb = [0, 0, 0]    # 决策变量下界
        ub = [1, 1, 2]    
        lbin = [1, 1, 0]  # 决策变量下界类型，1包含边界，0不包含
        ubin = [1, 1, 0]  
        ea.Problem.__init__(self, name, M, maxormins, Dim,
                            varTypes, lb, ub, lbin, ubin)

    def aimFunc(self, pop):
        """目标函数
            @pop：传入的种群对象
        """
        Vars = pop.Phen    # 决策变量矩阵
        x1 = Vars[:, [0]]  # 变量x1组成的列向量
        x2 = Vars[:, [1]]  
        x3 = Vars[:, [2]]
        
        pop.ObjV = 4*x1 + 2*x2 + x3 # 目标函数值
        # 采用可行性法则处理约束，生成种群个体违反约束程度矩阵
        pop.CV = np.hstack([2*x1 + x2 - 1,  # 第一个约束
                            x1 + 2*x3 - 2,  # 第二个约束
                            np.abs(x1 + x2 + x3 - 1)])  # 第三个约束
```

本案例的问题包含不等式约束和等式约束，在Geatpy中有两种处理约束条件的方法：罚函数法和可行性法则。

上面的代码展示的即为采用可行性法则处理约束的用法，它需要计算每个个体违反约束的程度，并把结果保存在种群类的`CV`矩阵中。`CV`矩阵的行数等于种群个体数（每一行对应一个个体）、列数等于约束条件数量（每一列对应一个约束条件，可以是等式约束也可以是不等式约束）。CV 矩阵中元素小于或等于0表示对应个体满足对应的约束条件，否则是违反对应的约束条件，大于0的值越大，表示违反约束的程度越高。

> 这里等式约束$x_{1}+x_{2}+x_{3}=1$转换成了绝对值$|x_{1}+x_{2}+x_{3}-1|$，也就是两边相等时候为0，不相等时差值越大约束也就越高。

若要使用罚函数法，则不需要生成`CV`矩阵，最简单的方法是利用`Numpy`的`where`语句把违反约束条件的个体索引找到，并根据该索引对种群的对应位置上的目标函数值加以惩罚即可。但本案例中包含一个**等式约束**，用这种简单的惩罚方法难以找到可行解。若要采用这种方法，目标函数“aimFunc()”可如下定义：

```script magic_args="true"

def aimFunc(self, pop):
    """采用罚函数法处理约束"""
    Vars = pop.Phen    # 决策变量矩阵
    x1 = Vars[:, [0]]
    x2 = Vars[:, [1]]
    x3 = Vars[:, [2]]
    f = 4*x1 + 2*x2 + x3  # 计算目标函数值
    exIdx1 = np.where(2*x1+x2 > 1)[0]    # 找到违反约束条件1的索引
    exIdx2 = np.where(x1+2*x3 > 2)[0]    # 找到违反约束条件2的索引
    exIdx3 = np.where(x1+x2+x3 != 1)[0]  # 找到违反约束条件3的索引
    exIdx = np.unique(np.hstack([exIdx1, exIdx2, exIdx3]))  # 合并索引
    alpha = 2  # 惩罚缩放因子
    beta = 1   # 惩罚最小偏移量w
    f[exIdx] += self.maxormins[0]*alpha * (np.max(f)-np.min(f)+beta)
    pop.ObjV = f
```

### 迭代求解

```python
import numpy as np
import geatpy as ea # import geatpy

problem = MyProblem() # 实例化问题对象

"""种群设置"""
Encoding = 'RI' # 编码方式
NIND = 50 # 种群规模
Field = ea.crtfld(Encoding, problem.varTypes, problem.ranges,
problem.borders) # 创建区域描述器
population = ea.Population(Encoding, Field, NIND)

"""算法参数设置"""
myAlgorithm = ea.soea_DE_best_1_L_templet(problem, population)
myAlgorithm.MAXGEN = 1000 # 最大遗传代数
myAlgorithm.mutOper.F = 0.5 # 设置差分进化的变异缩放因子
myAlgorithm.recOper.XOVR = 0.5 # 设置交叉概率
myAlgorithm.drawing = 1 # 0关闭，1绘制结果图，2绘制动态过程图

[population, obj_trace, var_trace] = myAlgorithm.run()
```

```python
best_gen = np.argmax(obj_trace[:, 1])
best_ObjV = obj_trace[best_gen, 1]

print('最优的目标函数值为： %s'%(best_ObjV))
print('最优的决策变量值为： ')
for i in range(var_trace.shape[1]):
    print(var_trace[best_gen, i])
print('有效进化代数： %s'%(obj_trace.shape[0]))
print('最优的一代是第 %s 代'%(best_gen + 1))
print('评价次数： %s'%(myAlgorithm.evalsNum))
print('时间已过 %s 秒'%(myAlgorithm.passTime))
```

## 带约束的多目标优化
下面看一个带约束的多目标优化问题：

$$\min =\left\{\begin{array}{l}
f_{1}(x, y)=4 x^{2}+4 y^{2} \\
f_{2}(x, y)=4(x-5)^{2}+4(y-5)^{2}
\end{array}\right.$$

$$\text {s.t.}=\left\{\begin{array}{c}
g_{1}(x, y)=(x-5)^{2}+y^{2} \leq 25 \\
g_{2}(x, y)=(x-8)^{2}+(y-3)^{2} \geq 7.7 \\
0 \leq x \leq 5,0 \leq y \leq 3
\end{array}\right.$$

```python
import numpy as np
import geatpy as ea


class MyProblem(ea.Problem):
    def __init__(self):
        name = 'BNH'
        M = 2  # 目标维数
        maxormins = [1] * M  # 目标最小/最大化，1:min；-1:max
        Dim = 2  # 决策变量维数
        varTypes = [0] * Dim  # 决策变量的类型，0实数；1整数
        lb = [0] * Dim  # 决策变量下界
        ub = [5, 3]
        lbin = [1] * Dim  # 决策变量下边界
        ubin = [1] * Dim
        # 调用父类构造方法完成实例化
        ea.Problem.__init__(self, name, M, maxormins, Dim,
                            varTypes, lb, ub, lbin, ubin)

    def aimFunc(self, pop):
        Vars = pop.Phen  # 决策变量矩阵
        x1 = Vars[:, [0]]
        x2 = Vars[:, [1]]
        f1 = 4*x1**2 + 4*x2**2
        f2 = (x1 - 5)**2 + (x2 - 5)**2
        # 采用可行性法则处理约束
        pop.CV = np.hstack([(x1 - 5)**2 + x2**2 - 25, -
                            (x1 - 8)**2 - (x2 - 3)**2 + 7.7])
        pop.ObjV = np.hstack([f1, f2]) # 目标函数值

    def calReferObjV(self):
        """计算全局最优解"""
        N = 10  # 欲得到10000个真实前沿点
        x1 = np.linspace(0, 5, N)
        x2 = x1.copy()
        x2[x1 >= 3] = 3
        return np.vstack((4 * x1**2 + 4 * x2**2, (x1 - 5)**2 + (x2 - 5)**2)).T
```

```python
problem = MyProblem()

"""种群设置"""
Encoding = 'RI'  # 实整数编码
NIND = 100  # 种群规模
Field = ea.crtfld(Encoding, problem.varTypes,
                  problem.ranges, problem.borders)  # 创建区域描述器
population = ea.Population(Encoding, Field, NIND)

"""算法参数设置"""
myAlgorithm = ea.moea_NSGA2_templet(problem, population)
myAlgorithm.MAXGEN = 200  # 最大遗传代数
myAlgorithm.drawing = 1  # 绘图方式

"""调用算法模板进行种群进化
调用run执行算法模板， 得到帕累托最优解集NDSet。
NDSet是一个种群类Population的对象。
NDSet.ObjV为最优解个体的目标函数值； NDSet.Phen为对应的决策变量值。
详见Population.py中关于种群类的定义。
"""
NDSet = myAlgorithm.run() # 执行算法模板， 得到非支配种群
#NDSet.save() # 把结果保存到文件中
```

```python
# 计算指标
PF = problem.getReferObjV() # 获取真实前沿
if PF is not None and NDSet.sizes != 0:
    GD = ea.indicator.GD(NDSet.ObjV, PF) # 计算GD指标
    IGD = ea.indicator.IGD(NDSet.ObjV, PF) # 计算IGD指标
    HV = ea.indicator.HV(NDSet.ObjV, PF) # 计算HV指标
    Spacing = ea.indicator.Spacing(NDSet.ObjV) # 计算Spacing指标
    print('GD: %f'%GD)
    print('IGD: %f'%IGD)
    print('HV: %f'%HV)
    print('Spacing: %f'%Spacing)
```
