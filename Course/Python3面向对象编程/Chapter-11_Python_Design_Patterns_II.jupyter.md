# 第11章 Python设计模式II
## 适配器模式
在结构上，适配器模式和简化版的装饰器模式很像。装饰器提供与被它替换对象同样的接口，而适配器在两个不同的接口之间进行映射。下面是UML图

![](http://static.zybuluo.com/AustinMxnet/ayw1tdoqnqe2cvp7nxdc4hwx/image.png)

`Interface1`会调用名为`make_action(some, arguments)`的方法，然而我们也有一个相当完美的`Interface2`类可以完成所有我们想要做的事，只是提供了名为`different_action(other, arguments)`的方法。`Adapter`类实现了`make_action()`方法，且将参数映射到已有的接口上。

适配器的优势在于从一个接口映射到另一个接口**全部在一个地方完成**。例如，我们有下面这样一个类，接受格式为`YYYY-MM-DD`的日期字符串并计算一个人在某一天的年龄：

```python
class AgeCalculator:
    def __init__(self, birthday):
        self.year, self.month, self.day = (
            int(x) for x in birthday.split('-'))

    def calculate_age(self, date):
        year, month, day = (
            int(x) for x in date.split('-'))
        age = year - self.year
        if (month, day) < (self.month, self.day):
            age -= 1
        return age

    
born_date = '1990-2-21'
curren_date = '2001-3-20'
AgeCalculator(born_date).calculate_age(curren_date)
```

然后Python内置的`datetime`模块提供了各种日期和时间对象，另一个程序希望接收`datetime.date`的对象：

```python
import datetime

born_date = datetime.date(1990, 12, 22)
curren_date = datetime.date(2010, 12, 22)

print(born_date.strftime("%Y-%m-%d"))
print(born_date.strftime('%d-%m,%Y'))
```

我们有很多解决方法，例如重写`AgeCalculator`并接受`datetime`对象。但是如果这个类是由第三方库提供的，我们不知道或者不能修改其内部结构，那就需要尝试其他方法。我们还可以在每次计算`datetime.date`对象对应的年龄时，可以通过调用`datetime.date.strftime('%Y-%m-%d')`在传入`AgeCalculator`，但是这种转换每次都要写，将来所有地方都要维护，并且违反了DRY原则。

我们可以用适配器来模式（任何时候只要维护这个类即可）：

```python
class DateAgeAdapter:
    def _str_date(self, date):
        return date.strftime("%Y-%m-%d")

    def __init__(self, birthday):
        birthday = self._str_date(birthday)
        self.calculator = AgeCalculator(birthday)

    def get_age(self, date):
        date = self._str_date(date)
        return self.calculator.calculate_age(date)

DateAgeAdapter(born_date).get_age(curren_date)
```

## 门面模式
## 享元模式
享元模式是一种内存优化的设计模式。注意Python程序员往往会忽略内存优化，并默认为内置的垃圾回收器将会处理这些问题。通常来说这是没有问题的；不过当我们需要开发大型应用，其中包含许多彼此相关的对象时，注意考虑内存问题将会带来很大收益。

![](http://static.zybuluo.com/AustinMxnet/2y9ipzsiketncg407neuwcak/image.png)

每个Flyweight都没有特定的状态，无论何时它需要执行`SpecificState`上的操作，都需要通过调用者传入或者已经传入的`Flyweight`对象。

考虑一个汽车销售的清单系统。每辆汽车都有一个特定的序列号和特定的颜色等，但是同一型号汽车的大部分配置（如长度、宽度、座位数等）都是相同的
。Honda一年销售汽车的数量很可观，如果对每辆车都单独存储这些共有信息将会浪费大量的内存空间。通过使用享元模式，我们可以让同一型号的汽车共用同一个特征对象，之后对于每辆车只需要引用型号特征和特定的序列号与颜色即可。

在Python中，享元对象的生产者通常通过`__new__`构造函数来实现，类似单例模式。但是**单例模式只需要返回一个实例，而享元模式需要根据键返回不同的实例**。

> 这里不对`weakref.WeakValueDictionary()`展开介绍。引用官方介绍：For example, if you have a number of large binary image objects, you may wish to associate a name with each. If you used a Python dictionary to map names to images, or images to names, the image objects would remain alive just because they appeared as values or keys in the dictionaries. The `WeakKeyDictionary` and `WeakValueDictionary` classes supplied by the `weakref` module are an alternative, using weak references to construct mappings that don't keep objects alive solely because they appear in the mapping objects. If, for example, an image object is a value in a `WeakValueDictionary`, then when the last remaining references to that image object are the weak references held by weak mappings, garbage collection can reclaim the object, and its corresponding entries in weak mappings are simply deleted.

```python
import weakref


class CarModel:
    _models = weakref.WeakValueDictionary()

    def __new__(cls, model_name, *args, **kwargs):
        model = cls._models.get(model_name)
        if not model:
            model = super().__new__(cls)
            cls._models[model_name] = model

        return model

    def __init__(self, model_name, air=False, tilt=False,
                 cruise_control=False, power_locks=False,
                 alloy_wheels=False, usb_charger=False):
        if not hasattr(self, "initted"):
            self.model_name = model_name
            self.air = air
            self.tilt = tilt
            self.cruise_control = cruise_control
            self.power_locks = power_locks
            self.alloy_wheels = alloy_wheels
            self.usb_charger = usb_charger
            self.initted = True

    def check_serial(self, serial_number):
        print("Sorry, we are unable to check "
              "the serial number {0} on the {1} "
              "at this time".format(serial_number, self.model_name))
```

```python
class Car:
    def __init__(self, model, color, serial):
        self.model = model
        self.color = color
        self.serial = serial

    def check_serial(self):
        return self.model.check_serial(self.serial)
```

```python
dx = CarModel("FIT DX")
lx = CarModel("FIT LX", air=True)
lx1 = CarModel("FIT LX")

Car1 = Car(dx, "blue", "12345")
Car2 = Car(dx, "red", "12346")
Car3 = Car(lx, "yellow", "12347")
```

`lx1`的得到的对象其实还是`lx`:

```python
id(lx) == id(lx1)
```

## 命令模式
命令模式在必须执行的动作与调用这些动作的对象之间增加了一个抽象层。在命令模式中，客户端创建一个`Command`对象并且稍后执行，这个对象知道接收对象`Receiver`在执行命令时能管理内部状态。`Command`对象实现一个特定的接口（通常是`execute`或`do_action`方法），并负责执行操作所需的所有参数。最后，一个或多个`Invoker`对象将会在正确的时间执行这一命令。

下面是UML图

![](http://static.zybuluo.com/AustinMxnet/0n8yzrv5rll9xbthgeopgoy1/image.png)

图形窗口是命令模式的一个例子，通过菜单选项、键盘快捷键、工具条上的图标或上下文菜单来触发操作。这些都是`Invoker`对象。真正发生的操作，例如`Exit`、`Save`或`Copy`则是`CommandInterface`的具体实现。GUI窗口可以接收退出操作，文档可以接收保存操作，ClipboardManager可以接收复制操作，这些都是可能的Receivers的例子。

先看一种简单的形式：

```python
import sys


class Window:
    def exit(self):
        # sys.exit(0)
        print('window is exiting...')


class MenuItem:
    def click(self):
        self.command()


window = Window()
exit_menu = MenuItem()
exit_menu.command = window.exit

exit_menu.click()
```

如果有多个`window`对象，用上面的代码就需要多个`exit_menu`，如果menu很多，就会很难维护。我们可以考虑用一个`exit_menu`对应多个`window`对象：

```python
class Window:
    def __init__(self, name):
        self.name = name

    def exit(self):
        print("window '{}' is exiting...".format(self.name))


class MenuItem:
    def click(self, obj):
        self.command(obj)


window1 = Window('note1')
window2 = Window('note2')

exit_menu = MenuItem()
exit_menu.command = Window.exit
exit_menu.click(window1)
exit_menu.click(window2)
```

上面代码利用了类函数默认带有`self`参数的性质，由于不确定实例对象，所以我们将对应类的函数传给`command`对象，最后在运行时传入需要被执行的实例化对象。当然，我们可以`MenuItem`对象添加`__call__`方法，不需要限定为函数（如这里的`click`）。

前面的例子适用于不需要维护状态信息的场景，在更复杂的应用中，我们也可以采用下面这段代码：

```python
class Document:
    def __init__(self, filename):
        self.filename = filename
        self.contents = "This file cannot be modified"

    def save(self):
        print("Save '{0}' to '{1}'".format(self.contents, self.filename))


class KeyboardShortcut:
    def keypress(self):
        self.command()


class SaveCommand:
    def __init__(self, document):
        self.document = document

    def __call__(self):
        self.document.save()


document = Document("a_file.txt")
shortcut = KeyboardShortcut()
save_command = SaveCommand(document)
shortcut.command = save_command

shortcut.keypress()
save_command()
```

命令模式通常需要扩展支持撤销指令。例如，一个文本程序可能将每一次插入关联到一个不同的命令，这一命令不光有`execute`方法，还有一个`undo`方法用来删除刚刚插入的内容。一个绘图程序可能会将每一次绘制操作（长方形、线条、徒手绘画等）关联到一个命令，该命令拥有一个`undo`方法以重置像素到初始状态。在这些例子中，命令模式的解耦特性就显得更加有用了，因为每个动作都维护了足够的状态，以便用来撤销操作。

由于刚才的方法需要对每个命令都创建一个类（如刚才的`SaveCommand`），个人不是很喜欢，所以尝试用之前方法实现了一个带`undo`功能的文本编辑器：

```python
from functools import wraps


class DocText:
    def __init__(self):
        self.data = []
        self.cmdh = []  # command history
        self.__repr__ = self.__str__

    def __str__(self):
        return "".join((str(c) for c in self.data))

    def append(self, d, undo=False):
        if undo:
            del self.data[-len(d):]
        else:
            self.data += [i for i in d]


class Document:
    def __init__(self, filename):
        self.filename = filename
        self.doctext = DocText()

    def save(self):
        print("Save '{0}' to '{1}'".format(self.doctext, self.filename))

    def reload(self):
        print("Reload from '{}'".format(self.filename))


doc = Document('test.txt')
doc.doctext.append(d='I love you! ')
doc.doctext.append('Ship Girl')
doc.save()
print(doc.doctext)
```

利用Python的装饰器存储每次调用的函数和其参数，实现`undo`功能：

```python
from functools import wraps


class DocText:
    def __init__(self):
        self.data = []
        self.cmdh = [] # command history
        self.__repr__ = self.__str__

    def __str__(self):
        return "".join((str(c) for c in self.data))
    
    def record_cmd(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            self.cmdh.append((func, args, kwargs))
            return func(self, *args, **kwargs)
        return wrapper

    @record_cmd
    def append(self, d, undo=False):
        if undo:
            del self.data[-len(d):]
        else:
            self.data += [i for i in d]
            
    def undo(self):
        if len(self.cmdh) == 0:
            print("No history to undo!")
            return None
        func, args, kwargs = self.cmdh.pop()
        kwargs.update(undo=True)
        return func(self, *args, **kwargs)


class Document:
    def __init__(self, filename):
        self.filename = filename
        self.doctext = DocText()

    def save(self):
        print("Save '{0}' to '{1}'".format(self.doctext, self.filename))

    def reload(self):
        print("Reload from '{}'".format(self.filename))


doc = Document('test.txt')
doc.doctext.append(d='I love you! ')
doc.doctext.append('Ship Girl')
doc.save()
doc.doctext.undo() # 撤销上一次命令
print(doc.doctext)
doc.doctext.undo() # 撤销上一次命令
doc.doctext.undo() # 没有命令可以被撤销
```

```python
class DocMenu(dict):
    def new(self, name, cmd):
        self[name] = cmd


menu = DocMenu()
menu.new('Save', cmd=Document.save)
menu.new('Reload', cmd=Document.reload)

menu['Save'](doc)
reload_menu = menu['Reload']
reload_menu(doc)

menu
```

```python
class KeyboardShortcut:
    def __init__(self, key, cmd):
        print("Register {} to {}".format(key, cmd.__name__))
        self.key = key
        self.cmd = cmd

    def keypress(self, obj):
        self.cmd(obj)


undo_shortcut = KeyboardShortcut('ctrl+z', DocText.undo)
undo_shortcut.keypress(doc.doctext)
```

## 抽象工厂模式
当有多种配置需求或多种系统平台时，一般会用抽象工厂模式。调用者向抽象工厂请求对象时，并不知道返回的是哪一种类（虽然接口一样），其底层实现可能依赖于很多因素，例如当前时区、操作系统或本地的配置信息。

抽象工厂模式的一个常见例子就是跨平台的工具、数据库和不同国家特有的格式或计算器。我们创建一个具体的例子，对日期和货币进行格式化输出，需要根据不同的地域信息返回不同的格式化标准。首先创建一个抽象工厂类为不同国家选择不同的工厂，然后创建几个具体的工厂，其中一个用于法国标准，另一个用于美国标准，下面是UML图

![](http://static.zybuluo.com/AustinMxnet/km5p9oqy6cqfsskqq43l0opw/image.png)

两国日期和货币的格式如下：

| | 美国 | 法国 |
| :--: | :--------: | :--------: |
| 日期 | mm-ydd-yyy | dd/mm/yyyy |
| 货币 | $14,500.50 | 14 500€50 |

```python
class FranceDateFormatter:
    def format_date(self, y, m, d):
        y, m, d = (str(x) for x in (y, m, d))
        y = '20' + y if len(y) == 2 else y
        m = '0' + m if len(m) == 1 else m
        d = '0' + d if len(d) == 1 else d
        return("{0}/{1}/{2}".format(d, m, y))


class USADateFormatter:
    def format_date(self, y, m, d):
        y, m, d = (str(x) for x in (y, m, d))
        y = '20' + y if len(y) == 2 else y
        m = '0' + m if len(m) == 1 else m
        d = '0' + d if len(d) == 1 else d
        return("{0}-{1}-{2}".format(m, d, y))
```

```python
france_date = FranceDateFormatter()
usa_date = USADateFormatter()

print(france_date.format_date(2012, 12, 2))
print(usa_date.format_date(2012, 12, 2))
```

```python
class FranceCurrencyFormatter:
    def format_currency(self, base, cents):
        base, cents = (str(x) for x in (base, cents))
        if len(cents) == 0:
            cents = '00'
        elif len(cents) == 1:
            cents = '0' + cents

        digits = []
        for i, c in enumerate(reversed(base)):
            if i and not i % 3:
                digits.append(' ')
            digits.append(c)
        base = ''.join(reversed(digits))
        return "{0}€{1}".format(base, cents)


class USACurrencyFormatter:
    def format_currency(self, base, cents):
        base, cents = (str(x) for x in (base, cents))
        if len(cents) == 0:
            cents = '00'
        elif len(cents) == 1:
            cents = '0' + cents

        digits = []
        for i, c in enumerate(reversed(base)):
            if i and not i % 3:
                digits.append(',')
            digits.append(c)
        base = ''.join(reversed(digits))
        return "${0}.{1}".format(base, cents)
```

```python
france_currency = FranceCurrencyFormatter()
usa_currency = USACurrencyFormatter()

print(france_currency.format_currency(120000, 2))
print(usa_currency.format_currency(120000, 2))
```

有了格式化类，只需要创建格式化工厂：

```python
class USAFormatterFactory:
    def create_date_formatter(self):
        return USADateFormatter()

    def create_currency_formatter(self):
        return USACurrencyFormatter()


class FranceFormatterFactory:
    def create_date_formatter(self):
        return FranceDateFormatter()

    def create_currency_formatter(self):
        return FranceCurrencyFormatter()


country_code = "US"
factory_map = {"US": USAFormatterFactory,
               "FR": FranceFormatterFactory}
formatter_factory = factory_map.get(country_code)()

formatter_factory.create_date_formatter().format_date(2010, 2, 13)
```

<!-- #region -->
上面的代码似乎仍有很多在Python中不需要的代码。我们可以通过不同的模块来表示不同的工厂类型（例如：美国和法国），然后确保每个工厂采用正确的模块，其目录结构可能是这样的：
```
localize/
    __init__.py
    backends/
        __init__.py
        USA.py
    France.py
    …
```

这里的小技巧是，`localize`包的`__init__.py`可以包含一些逻辑代码来将需求导向正确的工厂，例如其中一种：
```python
from .backends import USA, France

if country_code == "US":
    current_backend = USA
```

## 复合模式
复合模式允许从简单的组件构建复杂的树状结构。这些组件称为复合对象，复合对象为容器对象，其中的内容可能是另一个复合对象。如果一个组件包含子组件，则类似于容器，否则类似于普通变量。

一般来说，复合对象中的每个组件都必须是叶节点（不能包含其他对象）或者是复合节点，关键是包含具有相同的接口，如下图所示

![](http://static.zybuluo.com/AustinMxnet/2m9x0k3e46tgk1s3lv4fgvw9/image.png)

这种模式虽然简单，但是可以组合成复杂的结构：

![](http://static.zybuluo.com/AustinMxnet/ejn9komfcpwd9bygb5kfvxo4/image.png)

复合模式适合用在文件/文件夹的树形结构中。不管树中的节点是文件还是文件夹，都可以作为移动、复制或删除等操作的对象。我们可以创建一个包含这些操作的组件接口（component interface），然后用复合对象（composite object）表示文件夹，用叶子节点（leaf node）表示文件。我们先利用Python鸭子类型来隐式地（分别）提供这一接口：
<!-- #endregion -->

```python
class Folder:
    def __init__(self, name):
        self.name = name
        self.children = {}

    def add_child(self, child):
        pass

    def move(self, new_path):
        pass

    def copy(self, new_path):
        pass

    def delete(self):
        pass


class File:
    def __init__(self, name, contents):
        self.name = name
        self.contents = contents

    def move(self, new_path):
        pass

    def copy(self, new_path):
        pass

    def delete(self):
        pass
```

将一些通用方法抽象到父类`Component`接口并改为一个基类：

```python
class Component:
    def __init__(self, name):
        self.name = name

    def move(self, new_path):
        new_folder = get_node(new_path)
        del self.parent.children[self.name]
        new_folder.children[self.name] = self
        self.parent = new_folder

    def delete(self):
        del self.parent.children[self.name]


class Folder(Component):
    def __init__(self, name):
        super().__init__(name)
        self.children = {}

    def add_child(self, child):
        child.parent = self
        self.children[child.name] = child

    def copy(self, new_path):
        pass


class File(Component):
    def __init__(self, name, contents):
        super().__init__(name)
        self.contents = contents

    def copy(self, new_path):
        pass
```

<!-- #region -->
我们为`Component`类创建了`move`和`delete`方法。但是要注意两点：

1. `move`方法用了一个模块层的`get_node`函数，该函数从一个**预先定义的根节点**开始查找目标节点。因为所有文件/文件夹的操作都都要基于一个根节点。

1. `move`访问了一个神奇的尚未定义的`parent`变量，这个变量是在添加子节点时候定义的，而且**添加子节点`add_child`只能在`Folder`对象中完成**
，所以`File`对象虽然有`parent`属性，但是我们并没有在`File`对象的代码里看到。


所有文件都会被添加到根节点，或者根节点的某个子节点。对于 move 方法，移动的目的地应该是已经存在的文件夹，否则将会出现错误。和许多技术书籍中的例子一样，错误处理的部分都被忽略了，以帮助我们集中注意力于当前考虑的准则。
首先让我们来设定这个神奇的parent变量，这发生在文件夹的add_child方法中：
<!-- #endregion -->

```python
root = Folder('')


def get_node(path):
    names = path.split('/')[1:]
    node = root
    for name in names:
        node = node.children[name]
    return node
```

```python
folder1 = Folder('folder1')
folder2 = Folder('folder2')
root.add_child(folder1)
root.add_child(folder2)
folder11 = Folder('folder11')
folder1.add_child(folder11)
```

```python
get_node('').children
```
