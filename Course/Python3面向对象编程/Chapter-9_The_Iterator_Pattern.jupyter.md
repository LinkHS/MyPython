# 第9章 迭代器模式
**此章节内容很重要！！！**

## 迭代器
用设计模式的术语来说，迭代器是一个拥有`next()`和`done()`方法的对象，后者在序列迭代结束时返回`True`。在没有**内置支持迭代器**的编程语言中，迭代器的遍历过程看起来可能像这样：
```
while not iterator.done():
   item = iterator.next()
```

在Python中，迭代是一个特殊的特征，迭代对象`iterator`需要有`__next__()`方法。这个方法可以通过内置的`next(iterator)`访问。当遍历结束时，迭代器协议会抛出一个`StopIteration`异常，而不是通过`done`方法。

### 迭代器协议
Python的`collections.abc`模块中的抽象基类`Iterator`定义了迭代器协议。任何提供`__iter__()`方法的类都是可迭代的，这一方法必须返回一个`Iterator`实例（即实例化后的对象要有`__next__()`，用于每次迭代返回需要的值），通常`__iter__`函数通常返回它自己。

例如下面的示例中，`CapitalIterable`只有`__iter__()`但是返回了一个包含`__next__()`的对象`CapitalIterator`：

```python
class CapitalIterable:
    def __init__(self, string):
        self.string = string

    def __iter__(self):
        return CapitalIterator(self.string)


class CapitalIterator:
    def __init__(self, string):
        self.words = [w.capitalize() for w in string.split()]
        self.index = 0

    def __next__(self):
        if self.index == len(self.words):
            raise StopIteration()

        word = self.words[self.index]
        self.index += 1
        return word

    def __iter__(self):
        return self
```

```python
for i in CapitalIterable('hello world !'):
    print(i)
```

***重点！！！***  
用`yield`实现一个可迭代的函数会更简单。


## 列表推导
列表推导是通过高度优化的C代码实现的，在遍历大量元素时列表推导比`for`循环快得多。如果只是可读性还不足以说服你尽可能使用它，那么速度的提升应该可以做到这一点。

```python
odd_integers = {i for i in range(10) if i%2}
odd_integers
```

```python
coordinates = {i:j for i in range(5)
                   for j in range(5)
                   if i == j}
coordinates
```

***重点！！！***  
列表推导会将所有结果存储在容器中，当处理的信息量很大时，会比较占内存，可以用生成器表达式。


## 生成器表达式
有时候我们只希望处理一个序列，而不需要将一个新的列表、集合或字典放到系统内存中。例如每次遍历一个元素，就不浪费内存（尤其是数据量很大时），这时就要用到生成器表达式：

```python
odd_integers = (i for i in range(10) if i%2)
odd_integers
```

此时`odd_integers`没有任何结果产生，变成了一个可迭代的对象：

```python
for i in odd_integers:
    print(i)
```

### e.g. `WarningFilter`
我们想要从下面的日志文件中删除WARNING行：

```python
logs = [
"Jan 26, 2010 11:25:25  DEBUG       This is a debugging message.",
"Jan 26, 2010 11:25:36  INFO        This is an information method.",
"Jan 26, 2010 11:25:46  WARNING     This is a warning. It could be serious.",
"Jan 26, 2010 11:25:52  WARNING     Another warning sent.",
"Jan 26, 2010 11:25:59  INFO        Here's some information.",
"Jan 26, 2010 11:26:13  DEBUG       Debug messages are only useful if you want to figure something out.",
"Jan 26, 2010 11:26:32  INFO        Information is usually harmless, but helpful.",
"Jan 26, 2010 11:26:40  WARNING     Warnings should be heeded.",
"Jan 26, 2010 11:26:54  WARNING     Watch for warnings.",
]
```

最简单的方法：

```python
for line in logs:
    if 'WARNING' not in line:
        # file.write(line)
        print(line)
```

这样的代码似乎可读性很高，但是仅仅几行代码就有这么多缩进，非常难看。**如果我们想要对每一行做些别的事，可能逻辑会变得很糟糕**。

先不加其他逻辑，用列表推导试一下：

```python
no_warning = [line for line in logs if 'WARNING' not in line]
for line in no_warning:
    # file.write(line)    
    print(line)
```

`no_warning`存储了所有的结果，如果日志很长，这会很消耗内存。我们可以用生成器表达式：

```python
no_warning = (line for line in logs if 'WARNING' not in line)
for line in no_warning:
    # file.write(line)    
    print(line)
```

现在来考虑一种面向对象解决方案，不用任何简写：

```python
class WarningFilter:
    def __init__(self, insequence):
        self.insequence = insequence
        self.index = 0

    def __iter__(self):
        return self

    def __next__(self):
        while True:
            if self.index == len(self.insequence):
                raise StopIteration()
                break

            ret = self.insequence[self.index]
            self.index += 1
            if 'WARNING' in ret:
                continue
            return ret
```

```python
filter = WarningFilter(logs)
for l in filter:
    print(l)
```

似乎更复杂了？别急，用`yield`试一下：

```python
def warnings_filter(insequence):
    for l in insequence:
        if 'WARNING' in l:
            continue
        yield l
```

```python
filter = warnings_filter(logs)
for l in filter:
    print(l)
```

这样，就可以把复杂的逻辑全部放在`warnings_filter`中，主程序专注于自己的事情，也方便后期维护。

## e.g. 基于`yield from`的`walk()`
我们考虑一个经典的计算机科学问题：遍历树。最常见的树形数据结构就是计算机的文件系统。我们首先来模拟UNIX文件系统中的目录和文件，然后用`yield from`来高效地遍历：

```python
class File:
    def __init__(self, name):
        self.name = name


class Folder(File):
    def __init__(self, name):
        super().__init__(name)
        self.children = []

root = Folder('')
etc = Folder('etc')
root.children.append(etc)
etc.children.append(File('passwd'))
etc.children.append(File('groups'))
httpd = Folder('httpd')
etc.children.append(httpd)
httpd.children.append(File('http.conf'))
var = Folder('var')
root.children.append(var)
log = Folder('log')
var.children.append(log)
log.children.append(File('messages'))
log.children.append(File('kernel'))
```

整个目录结构如下：
```
/
/etc/
/etc/passwd
/etc/groups
/etc/httpd/
/etc/httpd/http.conf
/var/
/var/log/
/var/log/messages
/var/log/kernel
```
我们可以用递归方式打印出来：

```python
def Walk(file, father=''):
    name = father + file.name
    if isinstance(file, Folder):
        print(name+'/')
        for c in file.children:
            Walk(c, father=name+'/')
    else:
        print(name)

Walk(root)
```

当遍历文件时候，为了避免大量的内存消耗，或者找到想要处理的文件就可以，我们可以用`yield`和`yield from`达到读一个处理一个的目的：

```python
def Walk(file, father=''):
    name = father + file.name
    if isinstance(file, Folder):
        yield name+'/'
        for c in file.children:
            yield from Walk(c, father=name+'/')
    else:
        yield name

list(Walk(root))
```

***重点！！！***  
如果把`yield from`替换成`yield`会怎样：

```python
def Walk(file, father=''):
    name = father + file.name
    if isinstance(file, Folder):
        yield name+'/'
        for c in file.children:
            yield Walk(c, father=name+'/')
    else:
        yield name

list(Walk(root))
```

从结果可以看出`yield Walk()`没有继续迭代下去，**而`yield from Walk()`是可以迭代`Walk()`中所有的`yield`或`__next__`结果**。

因为递归有一些缺陷，这里给出了非递归版本：

```python
def Walk(file, father=''):
    todo = [[file, father]]
    while len(todo):
        file, father = todo.pop()
        name = file.name
        if isinstance(file, Folder):
            father = father + name + '/'
            print(father)
            for c in file.children:
                todo.append([c, father])
        else:
            print(father + name)

Walk(root)
```

## 协程[Coroutines]
协程本质上就是一个线程，以前线程任务的切换是由操作系统控制的，遇到I/O自动切换，现在我们用协程的目的就是较少操作系统切换的开销（开关线程，创建寄存器、堆栈、互斥锁等），应用程序自己控制任务的切换。

`input = yiled output`的功能很强大，例如实现一个异步计分器：

```python
def tally():
    score = 0
    while True:
        increment = yield score
        score += increment

Houston = tally() # 休斯顿队
next(Houston) # 初始化
Houston.send(2) # 得2分
Houston.send(3) # 得3分
```

`input = yiled output`的语法让我们得以根据输入计算需要返回的输出。如果只有`yield`，生成器只能按提前安排好的设置返回值。


### e.g. Kernel Log
前面的例子可以很容易地通过几个整数变量和`x += increment`来实现。让我们来看第二个例子，协程真正能够帮上忙的地方。

Linux内核日志文件看起来类似下面的日志：

```python
log_file=[
"unrelated log messages",
"sd 0:0:0:0 Attached Disk Drive", 
"unrelated log messages",
"sd 0:0:0:0 (SERIAL=ZZ12345)",
"unrelated log messages",
"sd 0:0:0:0 [sda] Options",
"unrelated log messages",
"XFS ERROR [sda]",
"unrelated log messages",
"sd 2:0:0:1 Attached Disk Drive",
"unrelated log messages",
"sd 2:0:0:1 (SERIAL=ZZ67890)",
"unrelated log messages",
"sd 2:0:0:1 [sdb] Options",
"unrelated log messages",
"sd 3:0:1:8 Attached Disk Drive",
"unrelated log messages",
"sd 3:0:1:8 (SERIAL=WW11111)",
"unrelated log messages",
"sd 3:0:1:8 [sdc] Options",
"unrelated log messages",
"XFS ERROR [sdc]",
"unrelated log messages",
]
```

需要从日志尾部找带有`ERROR`的行，然后根据这行里的bus名称(如`[sdc]`)找到带有序列号Serial number（如`3:0:1:8`）的行。

我们用正则表达式提取有`ERROR`的行：

```python
import re

for line in log_file:
    ERROR_RE = 'XFS ERROR (\[sd[a-z]\])'
    match = re.match(ERROR_RE, line)
    if match:
        print(line)
```

我们可以用正则表达式识别每一行内容，但是在遍历所有行的过程中**不得不更换正则表达式**，因为当前要查找的信息会因上一条信息不同而不同。总共需要3种类型的正则表达式，而且顺序还有依赖性，可能需要用的状态机和大量的`if...else...`来解决。

但是如果用协程就简单的多：

```python
def match_regex(log, regex):
    for line in reversed(log):
        match = re.match(regex, line)
        if match:
            regex = yield match.groups()[0]

def get_serials(log):
    ERROR_RE = 'XFS ERROR (\[sd[a-z]\])'
    matcher = match_regex(log, ERROR_RE)
    device = next(matcher)
    while True:
        bus = matcher.send('(sd \S+) {}.*'.format(re.escape(device)))
        serial = matcher.send('{} \(SERIAL=([^)]*)\)'.format(bus))
        yield serial
        device = matcher.send(ERROR_RE)
```

可以看出两个函数分工非常明确：
- `match_regex()`只负责查找符合规则的语句，具体规则由`get_serials()`通过`matcher.sned()`传递。 
- `get_serials()`只负责分配规则，并返回找到的信息。`get_serials`不需要逻辑判断切换规则，只要按既定规则顺序就可以，**这要归功于`match_regex()`只会返回符合的信息**。

```python
for serial_number in get_serials(log_file):
    print(serial_number)
```

## e.g. K近邻判断颜色种类
本章节最后是个案例，用K近邻算法判断颜色种类的，时间关系不多介绍，只介绍两点：
1. 读取文件用到了`yield`方法，并且直接返回一个`tuple`

```python
def load_colors(filename):
    with open(filename) as dataset_file:
        lines = csv.reader(dataset_file)
        for line in lines:
            yield tuple(float(y) for y in line[0:3]), line[3]
```

2. 写文件用到了`input = yield`进行同步操作

```python
def write_results(filename="output.csv"):
    with open(filename, "w") as file:
        writer = csv.writer(file)
        while True:
            color, name = yield
            writer.writerow(list(color) + [name])
```
