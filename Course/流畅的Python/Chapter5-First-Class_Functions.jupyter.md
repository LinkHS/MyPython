# 一等函数
## 函数内省
列出常规对象没有而函数有的属性：

```python
class C:
    pass

obj = C()

def func():
    pass

sorted(set(dir(func)) - set(dir(obj)))
```

## 从定位参数到仅限关键字参数

### e.g. `tag`函数用于生成HTML标签

```python
def tag(name, *content, cls=None, **attrs):
    """Generate one or more HTML tags"""
    if cls is not None:
        attrs['class'] = cls
    if attrs:
        attr_str = ''.join(' %s="%s"' % (attr, value)
                           for attr, value
                           in sorted(attrs.items()))
    else:
        attr_str = ''
    if content:
        return '\n'.join('<%s%s>%s</%s>' %
                         (name, attr_str, c, name) for c in content)
    else:
        return '<%s%s />' % (name, attr_str)
```

传入单个定位参数，生成一个指定名称的空标签：

```python
tag('br')
```

第一个参数后面的任意个参数会被`*content`捕获，存入一个元组：

```python
tag('p', 'hello')
```

`tag`函数签名中没有明确指定名称的关键字参数会被`**attrs`捕获，存入一个字典：

```python
tag('p', 'hello', id=33)
```

`cls`参数只能作为关键字参数传入：

```python
print(tag('p', 'hello', 'world', cls='sidebar'))
```

调用`tag`函数时，即便第一个定位参数也能作为关键字参数传入：

```python
tag(content='testing', name="img")
```

在`my_tag`前面加上`**`，字典中的所有元素作为单个参数传入，同名键会绑定到对应的具名参数上，余下的则被`**attrs`捕获：

```python
my_tag = {'name': 'img',
          'title': 'Sunset Boulevard',
          'src': 'sunset.jpg',
          'cls': 'framed'}

tag(**my_tag)
```

## 获取关于参数的信息
### e.g. 在指定长度附近截断字符串的函数

```python
def clip(text, max_len=80):
    """在max_len前面或后面的第一个空格处截断文本"""
    end = None
    if len(text) > max_len:
        space_before = text.rfind(' ', 0, max_len)
        if space_before >= 0:
            end = space_before
        else:
            space_after = text.rfind(' ', max_len)
            if space_after >= 0:
                end = space_after
    if end is None:  # 没找到空格
        end = len(text)
    return text[:end].rstrip()
```

### e.g. 提取关于函数参数的信息

```python
clip.__defaults__
```

```python
clip.__code__
```

```python
clip.__code__.co_varnames
```

```python
clip.__code__.co_argcount
```

上述结果可以看出，这种组织信息的方式并不是最便利的。参数名称在`__code__.co_varnames`中， 不过里面还有函数定义体中创建的局部变量。因此，参数名称是前$N$个字符串，$N$的值由`__code__.co_argcount`确定。

顺便说一下，这里不包含前缀为`*`或`**`的变长参数。参数的默认值只能通过它们在`__defaults__`元组中的位置确定，因此要从后向前扫描才能把参数和默认值对应起来。在这个示例中`clip`函数有两个参数，`text`和`max_len`，其中一个有默认值，即80，因此它必然属于最后一个参数，即`max_len`。

幸好，我们有更好的方式——使用`inspect`模块，见下面的示例。


### e.g. 提取函数的签名

```python
from inspect import signature

sig = signature(clip)
sig
```

```python
for name, param in sig.parameters.items():
    print(param.kind, ':', name, '=', param.default)

str(sig)
```

`kind`属性的值是`_ParameterKind`类中的5个值之一：
- POSITIONAL_OR_KEYWORD
- VAR_POSITIONAL
- VAR_KEYWORD
- KEYWORD_ONLY
- POSITIONAL_ONLY

## 支持函数式编程的包
### `operator`模块
在函数式编程中，经常需要把算术运算符当作函数使用。例如，不使用递归计算阶乘。operator模块提供了多个算数运算符对应的函数，另外还有能代替`lambda`读取元素或对象属性的`itemgetter`和`attrgetter`（会自行创建函数，类似`lambda`）等。

**V.S. lambda**  
[stackoverflow](https://stackoverflow.com/questions/2705104/lambda-vs-operator-attrgetterxxx-as-a-sort-key-function)说`itemgetter`等是调用对象的内置函数，速度上更快，而`lamdba`通过外部函数调用。

下面是`operator`模块中定义的部分函数：

```python
import operator

print([name for name in dir(operator) if not name.startswith('_')])
```

以`i`开头、后面是另一个运算符的那些名称（如`iadd`、`iand`等），对应的是增量赋值运算符（如`+=`、`&=`等）。

#### e.g. 使用`reduce`函数和一个`lambda`计算阶乘

```python
from functools import reduce

def fact(n):
    return reduce(lambda a, b: a*b, range(1, n+1))

fact(5)
```

#### e.g. 使用`reduce`和`operator.mul`函数计算阶乘

```python
from functools import reduce
from operator import mul


def fact(n):
    return reduce(mul, range(1, n+1))


fact(5)
```

#### e.g. 使用`itemgetter`排序一个元组列表

```python
from operator import itemgetter
metro_data = [('Tokyo', 'JP', 36.933, (35.689722, 139.691667)),
              ('Delhi NCR', 'IN', 21.935, (28.613889, 77.208889)),
              ('Mexico City', 'MX', 20.142, (19.433333, -99.133333)),
              ('New York-Newark', 'US', 20.104, (40.808611, -74.020386)),
              ('Sao Paulo', 'BR', 19.649, (-23.547778, -46.635833)), ]

for city in sorted(metro_data, key=itemgetter(1)):
    print(city)
```

如果把多个参数传给`itemgetter`，它构建的函数会返回提取的值构成的元组：

```python
cc_name = itemgetter(1, 0)
for city in metro_data:
    print(cc_name(city))
```

#### e.g. 使用`attrgetter`处理`namedtuple`
`metro_data`同上

```python
from collections import namedtuple

LatLong = namedtuple('LatLong', 'lat long')
Metropolis = namedtuple('Metropolis', 'name cc pop coord')

metro_areas = [Metropolis(name, cc, pop, LatLong(lat, long))
               for name, cc, pop, (lat, long) in metro_data]
metro_areas
```

按照纬度排序城市列表：

```python
metro_areas[0].coord.lat
```

```python
from operator import attrgetter

# 定义一个attrgetter，获取name和嵌套的coord.lat
name_lat = attrgetter('name', 'coord.lat')

for city in sorted(metro_areas, key=attrgetter('coord.lat')):
    print(name_lat(city))
```

其实`lambda`也能达到同样的功能：

```python
for city in sorted(metro_areas, key=lambda x: x.coord.lat):
    print(name_lat(city))
```

#### e.g. 使用methodcaller冻结参数
`methodcaller`允许使用字符串方式调用对象并冻结参数：

```python
from operator import methodcaller

upcase = methodcaller('upper')
upcase('hello')
```

```python
hiphenate = methodcaller('replace', ' ', '-')
hiphenate('hello world')
```

### 使用`functools.partial`冻结参数
TBD
