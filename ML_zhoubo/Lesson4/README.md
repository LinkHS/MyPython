
---
# `if __name__ == '__main__':`
如果没有此代码，那么 import 此 py 文件时候会执行不在函数里的代码

---
# `from __future__ import print_function`
如果没有此代码，那么 print 在 python2 和 python3 里打印的内容可能不一样，例如：
```
print('\t', __name__)
```
python2 识别不了`'\t'`

---
# `#  -*- coding:utf-8 -*-`