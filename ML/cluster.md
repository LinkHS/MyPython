一维数组聚类（natural breaks optimization）
```
def cluster(lst, max_gap):
    grps = []
    grp = []
    for i in range(len(lst)):
        #print(i, len(lst)-1, grp)
        if len(grp) == 0:
            grp.append(lst[i])
            continue
        elif lst[i] - grp[0] < max_gap:
            pass
        else:
            grps.append(grp.copy())
            grp = []
        grp.append(lst[i])
        if i == len(lst) - 1:
            grps.append(grp)
    return grps

array = [60, 65, 70, 76, 76, 78, 80, 83, 86, 90, 91]
grps = cluster(array, 7)
print(grps)
>>>
[[60, 65], [70, 76, 76], [78, 80, 83], [86, 90, 91]]
```


```
from sklearn.cluster import MeanShift
import numpy as np

data = np.array([60, 62, 65, 70, 76, 76, 78, 80, 83, 86, 90, 91]).reshape(-1, 1)

clf=MeanShift(bandwidth=7)
predicted=clf.fit_predict(data)

print(predicted)
>>>
[2 2 2 2 0 0 0 0 0 1 1 1]
```

---
[一维数组的聚类](https://www.biaodianfu.com/clustering-on-a-one-dimensional-array.html)
方案一：采用K-Means对一维数据聚类
方案二：采用一维聚类方法Jenks Natural Breaks
方案三：核密度估计Kernel Density Estimation

---
[Mean-Shift聚类法简单介绍及Python实现](https://www.cnblogs.com/feffery/p/8596482.html)
