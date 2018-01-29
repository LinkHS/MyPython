- [1. Contents](#1-contents)
    - [1.1. 4.0.sqrt.py](#11-40sqrtpy)
    - [1.2. 4.3.calc_e.py](#12-43calcepy)
    - [1.3. 4.4.calc_sin.py](#13-44calcsinpy)
- [2. Notes](#2-notes)
    - [`if __name__ == '__main__':`](#if-name-main)
    - [`from __future__ import print_function`](#from-future-import-printfunction)
    - [`#  -*- coding:utf-8 -*-`](#codingutf-8)

---
# 1. Contents
## 1.1. 4.0.sqrt.py
This file implements a simple (but not very fast) method to compute $\sqrt{x}$.

## 1.2. 4.3.calc_e.py
This file implements a simple (but not very fast) method to compute $e^x$.

To do this, we need to think in another way, $x = k*ln2 + r,\ |r|\le0.5*ln2$

so we can get:

$$e^x = e^{k \times ln2 + r}$$
$$= 2^k \times e^r $$

The reason why we compute $e^r$ rather than $e^x$ is that $e^r$ has better convergence.

## 1.3. 4.4.calc_sin.py
This file implements a simple (but not very fast) method to compute $sin(x)$.

To do this, we need to think in another way, $x = k*2\pi + r$

so we can get:

$$sin(x) = sin(r)$$


---
# 2. Notes
##

---
## `if __name__ == '__main__':`
如果没有此代码，那么 import 此 py 文件时候会执行不在函数里的代码

---
## `from __future__ import print_function`
如果没有此代码，那么 print 在 python2 和 python3 里打印的内容可能不一样，例如：
```
print('\t', __name__)
```
python2 识别不了`'\t'`

---
## `#  -*- coding:utf-8 -*-`