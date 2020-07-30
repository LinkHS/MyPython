# 第5章 何时使用面向对象编程
找出对象是面向对象分析与编程中非常重要的任务。**对象既有数据又有行为**：如果只需要处理数据，那么通常更好的方法是存储在列表、集合、字典或其他的Python数据结构中；另一方面，如果只需要用到行为，不需要存储数据，那么简单的函数就足够了。

## 通过属性向类数据添加行为
自定义取值方法对于需要根据其他属性计算的属性也很有用。例如，我们可能想要计算一个整数列表的平均值：

```python
class AverageList(list):
    @property
    def average(self):
        return sum(self) / len(self)

a = AverageList([1, 2, 3, 4])
a.average
```

这个简单的类继承自`list`，因此可以**任意使用与列表相似的行为**。我们只是给类添加了一个`@property`，于是奇迹般地拥有了平均值属性。当然，这里也可以将其定义为一个方法，命名为`calculate_average()`，因为方法代表的是动作。但是用一个被称为`average`的`property`更合适，易写又易读。

## e.g. 文本编辑器
我们将模拟一个Document，用于文本编辑器或文字处理工具。

Python中的字符串是不可变的，在没有创建新字符串对象的情况下，无法插入或移除字符。如果用`str`类管理内容，会导致越来越多的`str`对象占据内存，直到Python的垃圾回收器在后台将它们清理掉。因此，我们将用字符列表代替字符串。

> 真正的文本编辑器通常使用基于二叉树的数据结构，被称为rope，来模拟文档内容。

`Document`类应该拥有哪些方法呢？对于一个文本文档，需要支持插入、删除和选中字符、剪切、复制、粘贴，以及保存或关闭文档等。看起来有大量的数据和行为需要考虑，所以把它们全都收归`Document`类是合理的。

一个相关问题是，这个类应该由一堆Python基本对象组成吗？例如`str`类型的文件名、`int`类型的光标位置和`list`存储的字符。这些要用单独定义的对象吗？是否也需要由类来表示？

我们先从最简单的`Document`类开始：

```python
class Document:
    def __init__(self):
        self.characters = []
        self.cursor = 0
        self.filename = ''

    def insert(self, character):
        self.characters.insert(self.cursor, character)
        self.cursor += 1

    def delete(self):
        del self.characters[self.cursor]

    def forward(self):
        self.cursor += 1

    def back(self):
        self.cursor -= 1

    def save(self):
        with open(self.filename, 'w') as f:
            f.write(''.join(self.characters))

    @property
    def string(self):
        return "".join((str(c) for c in self.characters))
```

这个版本里，我们将`cursor`作为属性放在`Document`中，可以将方向键关联到对应的方法上（例如`back`、`forward`），现在来测试一下：

```python
doc = Document()
doc.insert('a')
doc.insert('b')
doc.insert('c')
print("Create 'abc':", doc.string)

doc.back()
doc.delete()
print("Delete 'c':", doc.string)
```

***重点！！！***  
如果我们想要关联的不止方向键，还想要关联Home和End键呢？当然可以添加更多的方法到`Document`类中。**但是如果为所有可能的移动操作（按单词移动，按句子移动，PageUp，PageDown，跳到文件结尾，跳到空格前，等等）关联一个方法，这个类将会变得很庞大，可能最好将这些方法放到一个单独的对象中**。

因此，让我们把光标属性变成一个对象，可以知道并且能够控制自己所在的位置：

```python
class Cursor:
    def __init__(self, document):
        self.document = document
        self.position = 0

    def forward(self):
        self.position += 1

    def back(self):
        self.position -= 1

    def home(self):
        while self.document.characters[self.position-1] != '\n':
            self.position -= 1
            if self.position == 0:
                # Got to beginning of file before newline
                break

    def end(self):
        while self.position < len(self.document.characters) and \
                self.document.characters[self.position] != '\n':
            self.position += 1
```

注意到`Cursor`类以`Document`实例作为一个自身属性，从而可以获取文档字符列表的内容（以及未来扩展）。虽然后面我们也会让`Cursor`的实例作为`Document`的一个属性（两者交叉了），但是这并不违反规则。

根据`Cursor`类改动`Document`中`cursor`相关内容（例如去除了`forward()`和`back()`）：
> `insert()`中的改变见后面的`Character`类

```python
class Document:
    def __init__(self):
        self.characters = []
        self.cursor = Cursor(self)
        self.filename = ''

    def insert(self, character):
        self.characters.insert(self.cursor.position,
                               character)
        self.cursor.forward()

    def delete(self):
        del self.characters[self.cursor.position]

    def save(self):
        with open(self.filename, 'w') as f:
            f.write(''.join(self.characters))

    @property
    def string(self):
        return "".join((str(c) for c in self.characters))
```

现在如果我们要操作光标，就要用`doc.cursor.xxx()`：

```python
doc = Document()
doc.insert('a')
doc.insert('b')
doc.insert('c')
print("Create 'abc':", doc.string)

doc.cursor.home()
doc.insert('0')
print("Insert '0':", doc.string)
```

目前这个纯文本文档框架很容易扩展。现在需要扩展到富文本：文本可以添加加粗、下画线或斜体格式。有两种方式可以做到：
1. 插入“假的”字符当作指示符（如html语言）
2. 为每个字符添加对应的说明信息

如果我们选择第二种方式，很显然需要一个带有字符信息属性的`Character`类。每个字符有3个布尔值属性表示是否需要加粗、斜体或下画线。

***重点！！！***  
这个`Character`类需要任何方法吗？如果不需要，也许应该用Python众多的数据结构之一来实现，例如元组或命名元组可能就足够了。可能会想到对字符执行删除或复制的操作，但是这些都在`Document`中处理了，而且严格意义上应该属于修改字符列表。**有需要针对单个字符的操作吗？** 似乎找到一个，就是打印出或者在屏幕上画出这些富文本。也许你会说这个方法可以放在`Document`中，**但是未来想改变展示的媒介（换个初始化对象就行），或者增加其他格式，就需要频繁改动`Document`类**。

现在我们只要重写`Character`的`__str__`，就可以让它打印出任何我们想要的东西：

```python
class Character:
    def __init__(self, character, bold=False, italic=False, underline=False):
        assert len(character) == 1
        self.character = character
        self.bold = bold
        self.italic = italic
        self.underline = underline

    def __str__(self):
        bold = "*" if self.bold else ''
        italic = "/" if self.italic else ''
        underline = "_" if self.underline else ''
        return bold + italic + underline + self.character
```

```python
print(Character('c', bold=True))
print(Character('c', italic=True))
print(Character('c', italic=True, underline=True))
```

有了`Character`，需要对`Document`和`Cursor`几处进行简单的修改，例如`insert()`中添加对普通字符的装饰，也就是实例化`Character`：

```python
class Document:
    def __init__(self):
        self.characters = []
        self.cursor = Cursor(self)
        self.filename = ''

    def insert(self, character):
        if not hasattr(character, 'character'):
            character = Character(character)
        self.characters.insert(self.cursor.position, character)
        self.cursor.forward()

    def delete(self):
        del self.characters[self.cursor.position]

    def save(self):
        with open(self.filename, 'w') as f:
            f.write(''.join(self.characters))

    @property
    def string(self):
        return "".join((str(c) for c in self.characters))
```

***重点！！！***  
`insert()`对输入做了检查是否有`character`属性，如果不是则用`Character`实例化。然而用户完全有可能希望既不用`Character`也不用普通字符串，而是使用鸭子类型。这帮助我们的程序利用鸭子类型和多态的优点，只要一个对象拥有`character`属性，就可以被用于`Document`类。

这个一般检查非常有用，例如，做一个带有语法高亮的程序员编辑器，需要字符拥有额外的数据，如这个字符属于什么类型的语法标记。注意，如果会用到大量这样的检验，可能最好将其实现为一个带有合适的`__subclasshook__`的抽象基类`Character`。

最后，在`Cursor`中的`home()`和`end()`函数中检查换行符时，需要替换成`Character.character`，而不止是之前存储的字符串字符：

```python
class Cursor:
    def __init__(self, document):
        self.document = document
        self.position = 0

    def forward(self):
        self.position += 1

    def back(self):
        self.position -= 1

    def home(self):
        while self.document.characters[self.position-1].character != '\n':
            self.position -= 1
            if self.position == 0:
                # Got to beginning of file before newline
                break

    def end(self):
        while self.position < len(self.document.characters) and \
                self.document.characters[self.position].character != '\n':
            self.position += 1
```

```python
doc = Document()
doc.insert('a')
doc.insert(Character('b', bold=True))
doc.insert(Character('c', italic=True))
print("Create 'abc':", doc.string)

doc.cursor.home()
doc.insert('0')
print("Insert '0':", doc.string)
```
