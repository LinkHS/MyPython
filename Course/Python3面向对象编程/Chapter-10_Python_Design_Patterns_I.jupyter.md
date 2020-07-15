# 第10章 Python设计模式I
## 装饰器模式
下面的示例中，装饰器`log_calls`接受一个函数对象作为参数并返回一个新的函数对象：

```python
import time


def log_calls(func):
    def wrapper(*args, **kwargs):
        now = time.time()
        print("Calling {0} with {1} and {2}".format(
            func.__name__, args, kwargs))
        return_value = func(*args, **kwargs)
        print("Executed {0} in {1}ms".format(func.__name__, time.time() - now))
        return return_value
    return wrapper


def test1(a, b, c):
    print("\ttest1 called")


def test2(a, b):
    print("\ttest2 called")


def test3(a, b):
    print("\ttest3 called")
    time.sleep(1)


test1 = log_calls(test1)
test2 = log_calls(test2)
test3 = log_calls(test3)

test1(1, 2, 3)
test2(4, b=5)
test3(6, 7)
```

## 观察者模式
观察者模式在状态监控和事件处理的情况中很有用。用这一模式可以让指定的对象**被未知的一组动态“观察者”对象所监控**。

核心对象中的值无论何时被更改，都会通过调用`update()`方法让所有的观察者对象知道。当核心对象被更改时，每个观察者可能负责不同的任务；**核心对象不知道也不关心这些任务是什么，这些观察者彼此之间也是如此**。

UML图如下所示

![](http://static.zybuluo.com/AustinMxnet/tpj4qqu38056nycev59j4d5q/image.png)

核心对象首先提供`attach()`方法链接观察者（或者称注册register），需要通知这些观察者时调用`_update_observers()`方法，这个方法将会遍历所有的观察者对象并告知其发生的改变。在下面的例子中，通知时直接调用观察者对象必须实现的`__call__()`来处理更新。

### e.g. 仓库更新通知
当仓库的库存种类和数量有变化时，通知各个观察者：

```python
class Inventory:
    def __init__(self):
        self.observers = []
        self._product = None
        self._quantity = 0

    def attach(self, observer):
        self.observers.append(observer)

    @property
    def product(self):
        return self._product

    @product.setter
    def product(self, value):
        self._product = value
        self._update_observers()

    @property
    def quantity(self):
        return self._quantity

    @quantity.setter
    def quantity(self, value):
        self._quantity = value
        self._update_observers()

    def _update_observers(self):
        for observer in self.observers:
            observer()
```

可以看到，当`Inventory`的`product`和`quantity`属性发生变化时，会调用`_update_observers()`去调用各个观察者，所以观察者必须实现`__call__()`：

```python
class ConsoleObserver:
    def __init__(self, inventory):
        self.inventory = inventory

    def __call__(self):
        print(self.inventory.product)
        print(self.inventory.quantity)
```

```python
i = Inventory()

# 观察者1
c1 = ConsoleObserver(i)
i.attach(c1)

i.product = 'widget'
```

```python
# 观察者2
c2 = ConsoleObserver(i)
i.attach(c2)

i.quantity = 2
```

观察者模式将**被观察的代码和观察的代码分离开来**。如果不用这种模式，就不得不将代码放到属性中去处理不同的情况，例如输出日志、更新数据库或文件等。所有完成这些任务的代码都将混在被观察的对象中。维护这样一个对象将会是一场噩梦，而且在日后想要添加新的监控功能也是非常痛苦的。


## 策略模式
策略模式的权威例子是排序程序。这些年以来，大量的算法被发明用来对一系列对象进行排序。快速排序、合并排序以及堆排序，这些都是拥有不同特征的快速排序算法。

![](http://static.zybuluo.com/AustinMxnet/0bo8c85x6zmj4yctl74fhgdt/image.png)


## 状态模式
状态模式在结构上与策略模式很像，但目标却迥然不同。状态模式的目标是用于表示状态转换系统：很明显这一系统会因为特定对象处于不同的状态而产生特定的活动。

为了实现这一系统，我们需要一个管理器或者上下文类来提供状态切换的接口。这个类的内部包含当前状态的指针，每个状态都知道自己可以根据特定的行为转换为哪些其他状态。

因此我们需要两种类：上下文类以及多个状态类。上下文类包含了当前的状态，以及当前状态下的行为。上下文类中调用的不同状态类彼此之间是不可见的，就像黑箱一样在内部执行状态管理。下面是UML图。

![](http://static.zybuluo.com/AustinMxnet/kfygyfe94kwj9yr8s057eqhb/image.png)

为了说明状态模式，让我们写一个XML解析工具。

```python
book = """
<book>
    <author>Dusty Phillips</author>
    <publisher>Packt Publishing</publisher>
    <title>Python 3 Object Oriented Programming</title>
    <content>
        <chapter>
            <number>1</number>
            <title>Object Oriented Design</title>
        </chapter>
        <chapter>
            <number>2</number>
            <title>Objects In Python</title>
        </chapter>
    </content>
</book>
"""
```

```python
print(book)
```

上下文类将会是解析器本身，在考虑状态和解析器之前，首先考虑程序的输出。我们需要的是一个`Node`对象所组成的树，它需要知道所解析标签的名字，例如上面的`book`、`author`等。另外，由于是树形结构，因此也需要指向父节点的指针和一个按顺序排列的子节点列表。有些节点需要保存文本值，而另一些节点则不需要保存文本值：

```python
class Node:
    def __init__(self, tag_name, parent=None):
        self.parent = parent
        self.tag_name = tag_name
        self.children = []
        self.text=""

    def __str__(self):
        if self.text:
            return self.tag_name + ": " + self.text
        else:
            return self.tag_name
```

```python
print(Node('book'))

author = Node('author', parent=Node('book'))
author.text = 'Dusty Phillips'
print(author)
```

在状态之间进行切换可能会比较棘手，我们怎么知道下一个节点是开始标签、结束标签还是文本节点？我们可以为每个状态添加一点逻辑流程，不过更合理的做法是创建一个新的状态全权负责状态切换。通过分析XML语法，我们可以分为如下几种状态：
1. `FirstTag`：如`<book>`
2. `ChildNone`：负责状态切换，除了`FirstTag`的其他状态
3. `OpenTag`：如`<author>`、`<publisher>`等，并且负责新建一个`Node`
4. `Text`：如`Dusty Phillips`
5. `CloseTag`：如`</author>`、`</publisher>`等

![](http://static.zybuluo.com/AustinMxnet/wvvdkbdt5kp8etkycb2eusc9/image.png)

第一个状态是`FirstTag`，然后到`ChildNode`，`ChildNode`根据当前内容判断下一步的状态（`OpenTag`，`Text`，`CloseTag`），这三种状态处理完都是返回到`ChildNode`。**不过为了处理`Node`嵌套，`OpenTag`需要新建一个`Node`，并且将这个新`Node`添加到当前`Node`的`children`中；`CloseTag`在处理完之后要将当前的`Node`还原为`parent`**。

有了各个状态和保存方式`Node`类，那解析器`Parser`应该如何设计呢？**由于各个状态自己决定下一步的状态，所以`Parser`不需要知道下一步的状态是什么，只需要调用一个必须实现的公共接口`state.process()`**，各个状态通过这个接口返回剩余未处理的字节。

```python
class Parser:
    def __init__(self, parse_string):
        self.parse_string = parse_string
        self.root = None
        self.current_node = None
        self.state = FirstTag()

    def _process(self, remaining_string):
        remaining = self.state.process(remaining_string, self)
        if remaining:
            self._process(remaining)

    def start(self):
        self._process(self.parse_string)
```

**分析到现在，在结合下面各个状态的实现，发现这个示例的设计有几点缺陷：**
1. 各个状态直接控制`parser.state`、`parser.current_node`等，将这些值返回给`Parser`让其自己设置更好。
2. 各个状态返回的是剩余未处理的字节，如果都是硬拷贝，消耗巨大，而且有时候由于文本过大并不适合全部load到内存中。

其他设计模式也可以完成本示例：
1. 递归
2. 协程

```python
class FirstTag:
    def process(self, remaining_string, parser):
        i_start_tag = remaining_string.find('<')
        i_end_tag = remaining_string.find('>')
        tag_name = remaining_string[i_start_tag+1:i_end_tag]
        root = Node(tag_name)
        parser.root = parser.current_node = root
        parser.state = ChildNode()
        return remaining_string[i_end_tag+1:]


class ChildNode:
    def process(self, remaining_string, parser):
        stripped = remaining_string.strip()
        if stripped.startswith("</"):
            parser.state = CloseTag()
        elif stripped.startswith("<"):
            parser.state = OpenTag()
        else:
            parser.state = TextNode()
        return stripped


class OpenTag:
    def process(self, remaining_string, parser):
        i_start_tag = remaining_string.find('<')
        i_end_tag = remaining_string.find('>')
        tag_name = remaining_string[i_start_tag+1:i_end_tag]
        node = Node(tag_name, parser.current_node)
        parser.current_node.children.append(node)
        parser.current_node = node
        parser.state = ChildNode()
        return remaining_string[i_end_tag+1:]


class TextNode:
    def process(self, remaining_string, parser):
        i_start_tag = remaining_string.find('<')
        text = remaining_string[:i_start_tag]
        parser.current_node.text = text
        parser.state = ChildNode()
        return remaining_string[i_start_tag:]


class CloseTag:
    def process(self, remaining_string, parser):
        i_start_tag = remaining_string.find('<')
        i_end_tag = remaining_string.find('>')
        assert remaining_string[i_start_tag+1] == "/"
        tag_name = remaining_string[i_start_tag+2:i_end_tag]
        assert tag_name == parser.current_node.tag_name
        parser.current_node = parser.current_node.parent
        parser.state = ChildNode()
        return remaining_string[i_end_tag+1:].strip()
```

```python
contents = book
p = Parser(contents)
p.start()

nodes = [p.root]
while nodes:
    node = nodes.pop(0)
    print(node)
    nodes = node.children + nodes
```

下面尝试用协程来解决：
1. `Node`类不变
2. 状态分为3类，分别为带有/没有`children`的`FatherTag`/`ChildTag`，和配对`FatherTag`的`CloseTag`
3. `swither()`只负责状态切换，将需要运行的状态告诉`Parser`，然后`Parser`调用标准接口（参数一致）
4. 每个状态处理完后负责返回当前`base_node`，`ChildTag`还要将创建的`Node`添加到其父`Node.children`中；`FatherTag`要将创建的`Node`变为当前的`base_node`；`CloseTag`要将`base_node`还原为其父`Node`

```python
book = [
"<book>",
"    <author>Dusty Phillips</author>",
"    <publisher>Packt Publishing</publisher>",
"    <title>Python 3 Object Oriented Programming</title>",
"    <content>",
"        <chapter>",
"            <number>1</number>",
"            <title>Object Oriented Design</title>",
"        </chapter>",
"        <chapter>",
"            <number>2</number>",
"            <title>Objects In Python</title>",
"        </chapter>",
"    </content>",
"</book>",
]
```

```python
def ChildTag(content, base_node):
    """
    e.g. <number>1</number>
    """
    assert content[0] == '<' and content[-1] == '>'
    i_ss_tag = 0
    i_se_tag = content.find('>')
    i_es_tag = content.find('</')
    tag_name = content[i_ss_tag+1:i_se_tag]

    node = Node(tag_name, base_node)
    node.text = content[i_se_tag+1:i_es_tag]
    base_node.children.append(node)
    return base_node


def FatherTag(content, base_node):
    """
    e.g. <book>
    """
    assert content[0] == '<' and content[-1] == '>'
    i_ss_tag = 0
    i_se_tag = content.find('>')
    tag_name = content[i_ss_tag+1:i_se_tag]

    node = Node(tag_name, base_node)
    base_node.children.append(node)
    return node


def CloseTag(content, base_node):
    """
    e.g. </book>
    """
    assert content[0] == '<' and content[-1] == '>'
    return base_node.parent


def ErrorTag(self, content, base_node):
    print('Error:', content)


def swither():
    line = yield
    while line != None:
        if line.startswith("</"):
            line = yield CloseTag
        elif line.startswith("<"):
            if line.find('</') > -1:
                line = yield ChildTag
            else:
                line = yield FatherTag
        else:
            line = yield ErrorTag
    yield
```

```python
class Parser:
    def __init__(self, XML):
        self.XML = XML
        self.node = Node('XML', None)

    def start(self):
        base_node = self.node
        get_state = swither()
        get_state.send(None)
        for line in self.XML:
            line = line.strip()
            #print('line:', line)
            state = get_state.send(line)
            base_node = state(line, base_node)
            #print('base_node:', base_node)
        get_state.send(None)
        self.node = base_node
```

```python
parser = Parser(book)
parser.start()

nodes = [parser.node]
while nodes:
    node = nodes.pop(0)
    print(node)
    nodes = node.children + nodes
```

## 单例模式
Python中没有私有构造函数，不过为了实现这一点，它有更好的解决方案。我们可以用`__new__`这一类方法来确保只会创建一个实例：

```python
class OneOnly:
    _singleton = None

    def __new__(cls, *args, **kwargs):
        if not cls._singleton:
            cls._singleton = super(OneOnly, cls).__new__(
                cls, *args, **kwargs)
        return cls._singleton
```

两个实例化的id是一样的：

```python
id(OneOnly()), id(OneOnly())
```

## 模板模式

![](http://static.zybuluo.com/AustinMxnet/6jtj3yp0azv8fuhx1fzp2l09/image.png)

### e.g. 汽车销售报告
我们有两个常规任务需要执行：
- 展示所有新车销售情况，以逗号分隔将其打印到屏幕上。
- 输出所有销售人员及其销售总额信息，将其保存到文件中，以逗号分隔，该文件可以导入电子表格中。

似乎是两个完全不同的任务，不过它们之间有一些相同的特征：

1. 连接到数据库。
1. 构造新车或销售总额的查询语句。
1. 执行查询语句。
1. 将结果格式化为以逗号分隔的字符串。

```python
class QueryTemplate:
    def connect(self):
        pass
    def construct_query(self):
        pass
    def do_query(self):
        pass
    def format_results(self):
        pass
    def output_results(self):
        pass

    def process_format(self):
        self.connect()
        self.construct_query()
        self.do_query()
        self.format_results()
        self.output_results()
```

```python
class NewVehiclesQuery(QueryTemplate):
    def construct_query(self):
        print("select * from Sales where new='true'")

    def output_results(self):
        print("output_results:......")


class UserGrossQuery(QueryTemplate):
    def construct_query(self):
        print("select salesperson, sum(amt) from Sales group by salesperson")

    def output_results(self):
        print("gross_sales_{0}".format('20200715'))
```

```python
NewVehiclesQuery().process_format()
```

```python
UserGrossQuery().process_format()
```
