# 第3章 对象相似时
## 基本的继承 [Basic Inheritance]
### e.g. 联系人管理器
这个案例包含了**“扩展内置对象”**和**“重写和`super`”**等关键知识。

联系人管理器可以追踪多个人的名字和Email地址，先设计联系人的类：

```python
class Contact:
    def __init__(self, name, email):
        self.name = name
        self.email = email
```

```python
John = Contact("John", "john@email.com")
John.name, John.email
```

思考一个问题：如果某些联系人也是供货商，我们需要从他们那里下单，该如何处理？

我们可以为`class Contact`添加一个`order`的方法，但是家人、朋友、客户等联系人也有了这个方法，会被意外地下单。

于是我们创建一个`class Supplier`，继承`Contact`，但是拥有`order`方法：

```python
class Supplier(Contact):
    def order(self, order):
        print("Order '{0}' from {1}".format(order, self.name))
```

```python
AppleSupplier = Supplier('Apple', 'apple@emial.com')
AppleSupplier.order('apples')
```

**扩展内置对象**

联系人管理器至少需要满足“存储”、“插入”和“删除”联系人等功能，不需要重复造轮子，于是用Python自带的`list`来管理联系人：

```python
all_contacts = []
all_contacts.append(John)
all_contacts.append(AppleSupplier)
all_contacts
```

`list`功能很强大，但是有些特定需求无法满足，例如根据名字搜索联系人，于是我们可以继承`list`创建一个带有`search`方法的类：

```python
class ContactList(list):
    def search(self, name):
        """Return all contacts that contain the search value in their name."""
        matching_contacts = []
        for contact in self:
            if name in contact.name:
                matching_contacts.append(contact)
        return matching_contacts
```

```python
all_contacts = ContactList()
all_contacts.append(John)
all_contacts.append(AppleSupplier)
all_contacts.search('John')[0].email
```

<!-- #region -->
大多数内置类型都可以被扩展，例如`set`、`dict`、`file`、`str`，甚至数字类型`int`、`float`也可以。


**重写和`super`**

对于大部分联系人来说，我们只要存储“姓名”和“邮箱”就可以了，但是对于好朋友，我们还想添加一个电话号码，同样新建`class Friend`继承`Contact`，然后只需要调用`super`方法就可以完成超类`Contact`中的操作：
<!-- #endregion -->

```python
class Friend(Contact):
    def __init__(self, name, email, phone):
        super().__init__(name, email)
        self.phone = phone
```

## 多重继承 [Multiple Inheritance]
由于多重继承会带来很多问题，而且任何提供了正确接口的对象都可以在Python中互相使用，通常当多重继承作为一种可用方案时，可用鸭子类型来模拟其中一个超类，因此本文不准备详细介绍多重继承，可以参考原书。

## 多态 [Polymorphism]

### e.g. 多媒体播放器
多态继承可以简化设计。例如多媒体播放器加载Audio File然后Play，不同文件（".mp3"，".wma"，".ogg"等）的解压缩和提取音频文件的过程是不一样的，于是可以设计一个基类`AudioFile`，其子类`WavFile`、`MP3File`等都有`play()`方法。**多媒体播放器永远不需要知道指向的是哪个子类，只需要调用`play()`方法并多态地让对象自己处理实际播放过程**：

```python
class AudioFile:
    def __init__(self, filename):
        if not filename.endswith(self.ext):
            raise Exception("Invalid file format")
        self.filename = filename
```

**重点！！！**  
`AudioFile`的`__init__`中有个非常巧妙的设计，检查文件后缀和`self.ext`是否匹配。虽然本身没有`ext`变量，但是不妨碍它访问子类中的类变量，这就强制了所有子类在初始化时候检查文件是否匹配，而且没有在子类中重复实现代码。

```python
class MP3File(AudioFile):
    ext = "mp3"
    def play(self):
        print("playing {} as mp3".format(self.filename))

class WavFile(AudioFile):
    ext = "wav"
    def play(self):
        print("playing {} as wav".format(self.filename))

class OggFile(AudioFile):
    ext = "ogg"
    def play(self):
        print("playing {} as ogg".format(self.filename))
```

```python
mp3 = MP3File("myfile.mp3")
mp3.play()
```

文件后缀不匹配时：

```python
try:
    Ogg = OggFile("myfile.mp3")
except Exception as E:
    print("Exception: {}".format(type(E).__name__))
    print("Exception message: {}".format(E))
```

在很多面向对象的场景中，多态是使用继承关系最重要的原因之一。

## 抽象基类 [Abstract Base Classes]
### 鸭子类型 [Duck Typing]
任何提供了正确接口的对象都可以在Python中互换使用，如果所有共享的都是公共接口（没有私有属性和方法调用），那用鸭子类型就可以满足，用继承来共享代码的需求就会降低。

另外鸭子类型另一个有用的特征是，鸭子类型的对象只需要提供真正被访问的方法和属性，不需要提供所需对象的整个接口。例如如果只有对文件只读需求，就可以创建一个新的对象，只实现`read()`而不需要实现`write()`。

例如Python内置的`list`、`dict`等都属于容器`Container`类：

```python
from collections import Container

print(issubclass(list, Container), isinstance(list(), Container))
print(issubclass(dict, Container), isinstance(dict(), Container))
```

但是**并不需要继承`Container`就可以创建一个属于`Container`类的对象**。下面我们来实现一个能判断给定值是否属于奇数集合的容器：

```python
class OddContainer:
    def __contains__(self, x):
        if not isinstance(x, int) or not x%2:
            return False
        return True

print("2 is odd:", 2 in OddContainer())
print("3 is odd:", 3 in OddContainer())
```

```python
isinstance(OddContainer(), Container), issubclass(OddContainer, Container)
```

为什么会这样呢？我们看一下`Container`的抽象方法：

```python
Container.__abstractmethods__
```

可以看出如果继承`Container`只需要实现一个抽象方法`__contains__`，再看下这个方法需要的参数：

```python
help(Container.__contains__)
```

很明显，这个参数就是需要被检查是否在这个容器中的值。

**从结果看出，`OddContainer`并没有继承`Container`，但是本身属于子类，并且实例化的对象也是一个`Container`的对象**。这就是鸭子类型比传统多态更实用的原因，可以避免继承甚至多重继承。

下面是另一个示例：

```python
from collections import Iterable
from collections import Iterator

class MyIterator:
    def __iter__(self):
        pass

    def __next__(self):
        pass
```

```python
print(issubclass(MyIterator, Iterable))
print(issubclass(MyIterator, Iterator))
print(isinstance(MyIterator(), Iterable))
print(isinstance(MyIterator(), Iterator))
```

### 创建抽象基类
虽然并不一定需要抽象基类才能用鸭子类型，但是为了避免使用者出错，我们需要强制子类实现某些方法或者属性，例如之前的多媒体播放器，其各个子类需要实现各自的`play()`方法和文件类型`ext`：

```python
import abc


class MediaLoader(metaclass=abc.ABCMeta):
    @abc.abstractclassmethod
    def play(self):
        pass

    @abc.abstractproperty
    def ext(self):
        pass

    @classmethod
    def __subclasshoo__(cls, C):
        if cls is MediaLoader:
            attrs = set(dir(C))
            if set(cls.__abstractmethods__) <= attrs:
                return True
        return NotImplemented
```

如果继承了`MediaLoader`而没实现`ext()`和`play()`就不能被实例化：

```python
class Wav(MediaLoader):
    pass

try:
    Wav()
except Exception as E:
    print("Exception: {}".format(type(E).__name__))
    print("Exception message: {}".format(E))
```

需要注意的是，`__subclasshoo__`用`#classmethod`装饰器包装成了一个类方法，用于检查`C`类是否是子类。注意，它不会检查`C`是否实现了这些方法，而检查是否存在，**因此`C`类如果是一个子类，但仍然可能是一个抽象类**。

这时我们再用之前的非继承方法就不可取了（但是还可以使用，就是无法通过`isinstance`的检测）：

```python
class Ogg:
    ext = '.ogg'
    def play(self):
        pass

print(issubclass(Ogg, MediaLoader), isinstance(Ogg(), MediaLoader))
```

必须继承`MediaLoader`：

```python
class Ogg(MediaLoader):
    ext = '.ogg'
    def play(self):
        pass
```

```python
print(issubclass(Ogg, MediaLoader), isinstance(Ogg(), MediaLoader))
```

### e.g. 房地产
我们将设计一个简单的房地产应用，允许经纪人管理可出售或出租的房产，房产有两类：Apartment和House。Agent应该可以输入与Property相关的信息，列出当前空闲的Porperty。

![](http://static.zybuluo.com/AustinMxnet/9lnem8s28f82wgvqzyuz0suz/image.png)

#### 设计`Property`类
虽然`House`和`Apartment`是两类房产，但是有一些共同的属性，例如面积、卧室数量和洗手间数量等，因此`Property`可以作为它们的超类。

***重点！！！***  
`prompt_init`让用户根据提示输入正确的房产信息，这些信息作为实例化`Property`各种子类的参数（包括父类共有参数和子类独有参数），简单示例如下，详情见`Agent.add_property()`。
```
init_args = PropertyClass.prompt_init()
PropertyClass(**init_args)
```

```python
def my_input(msg, inp='0'):
    """jupyter notebook自动化测试中代替内置`input()`
    """
    print("%s: %s" % (msg, inp))
    return inp


class Property:
    def __init__(self, square_feet='', beds='', baths='', **kwargs):
        super().__init__(**kwargs)
        self.square_feet = square_feet
        self.num_bedrooms = beds
        self.num_baths = baths

    def display(self):
        print("PROPERTY DETAILS")
        print("================")
        print("square footage: {}".format(self.square_feet))
        print("bedrooms: {}".format(self.num_bedrooms))
        print("bathrooms: {}".format(self.num_baths))
        print()

    @classmethod
    def prompt_init(cls):
        # return dict(square_feet=input("Enter the square feet: "),
        #            beds=input("Enter number of bedrooms: ", ),
        #            baths=input("Enter number of baths: "))
        return dict(square_feet=my_input("Enter the square feet: ", 100),
                    beds=my_input("Enter number of bedrooms: ", 3),
                    baths=my_input("Enter number of baths: ", 2))
```

```python
Property.prompt_init()
```

***重点！！！***  
因为我们知道`Property`将用于多重继承，所以给`__init__`添加了额外的`**kwargs`参数。并且也调用了`super().__init__`，**以防它不是在继承链的最后一层被调用**（这里的`super().__init__`并不是因为`Property`有父类，而是其子类会有多重继承，详情见附录A）。

#### 设计`House`和`Apartment`类
`Apartment`和`House`继承`Property`，并添加独有的属性和方法。`Apartment`需要说明是否有阳台、是否有洗衣房等，这些信息是需要用户输入的：

```python
def get_valid_input(input_string, valid_options):
    input_string += " ({}) ".format(", ".join(valid_options))
    #response = input(input_string)
    response = my_input(input_string, valid_options[0])
    while response.lower() not in valid_options:
        response = input(input_string)
    return response
```

```python
get_valid_input('Which laundry?', ("coin", "ensuite", "none"))
```

`get_valid_input()`需要被多处调用，而且其他类无关，所以作为模块层的函数使用。

```python
class Apartment(Property):
    valid_laundries = ("coin", "ensuite", "none")
    valid_balconies = ("yes", "no", "solarium")

    def __init__(self, balcony='', laundry='', **kwargs):
        super().__init__(**kwargs)
        self.balcony = balcony
        self.laundry = laundry

    def display(self):
        super().display()
        print("APARTMENT DETAILS")
        print("laundry: {}".format(self.laundry))
        print("has balcony: {}".format(self.balcony))

    @classmethod
    def prompt_init(cls):
        parent_init = super().prompt_init()
        laundry = get_valid_input("Laundry", cls.valid_laundries)
        balcony = get_valid_input("Balcony", cls.valid_balconies)
        parent_init.update({
            "laundry": laundry,
            "balcony": balcony
        })
        return parent_init
```

```python
apartment = Apartment(**Apartment.prompt_init())
print('\n')
apartment.display()
```

```python
class House(Property):
    valid_garage = ("attached", "detached", "none")
    valid_fenced = ("yes", "no")

    def __init__(self, num_stories='',
                 garage='', fenced='', **kwargs):
        super().__init__(**kwargs)
        self.garage = garage
        self.fenced = fenced
        self.num_stories = num_stories

    def display(self):
        super().display()
        print("HOUSE DETAILS")
        print("# of stories: {}".format(self.num_stories))
        print("garage: {}".format(self.garage))
        print("fenced yard: {}".format(self.fenced))

    @classmethod
    def prompt_init(cls):
        parent_init = super().prompt_init()
        fenced = get_valid_input("Is the yard fenced? ", cls.valid_fenced)
        garage = get_valid_input("Is there a garage? ", cls.valid_garage)
        #num_stories = input("How many stories? ")
        num_stories = my_input("How many stories?", '20')

        parent_init.update({
            "fenced": fenced,
            "garage": garage,
            "num_stories": num_stories
        })
        return parent_init
```

```python
house = House(**House.prompt_init())
print('\n')
house.display()
```

#### 设计`Purchase`和`Rental`类
我们继续探讨`Purchase`和`Rental`类。除了目的明显不同之外，它们与刚刚讨论过的几个类在设计上也很相似。

***重点！！！***  
因为我们是按照多重继承来设计的，所以我们知道这两个类会被子类以多重继承的方式继承（如`class HouseRental(Rental, House)`），并且顺序在前，所以我们在`display()`中加了`super().display()`，这样会导致这两个类的实例化无法调用`display()`，因为其父类`object`没有此方法。

```python
class Purchase:
    def __init__(self, price='', taxes='', **kwargs):
        super().__init__(**kwargs)
        self.price = price
        self.taxes = taxes

    def display(self):
        super().display()
        print("PURCHASE DETAILS")
        print("selling price: {}".format(self.price))
        print("estimated taxes: {}".format(self.taxes))

    def prompt_init():
        return dict(
            #price=input("What is the selling price? "),
            #taxes=input("What are the estimated taxes? "))
            price=my_input("What is the selling price?", "100"),
            taxes=my_input("What are the estimated taxes?", "20%"))
    prompt_init = staticmethod(prompt_init)


class Rental:
    def __init__(self, furnished='', utilities='',
                 rent='', **kwargs):
        super().__init__(**kwargs)
        self.furnished = furnished
        self.rent = rent
        self.utilities = utilities

    def display(self):
        super().display()
        print("RENTAL DETAILS")
        print("rent: {}".format(self.rent))
        print("estimated utilities: {}".format(self.utilities))
        print("furnished: {}".format(self.furnished))

    def prompt_init():
        return dict(
            #rent=input("What is the monthly rent? "),
            #utilities=input("What are the estimated utilities? "),
            rent=my_input("What is the monthly rent?", "12"),
            utilities=my_input("What are the estimated utilities?", "bed"),
            furnished=get_valid_input("Is the property furnished? ", ("yes", "no")))
    prompt_init = staticmethod(prompt_init)
```

```python
print(Purchase.prompt_init())
print('')
print(Rental.prompt_init())
```

#### 组合`House/Apartment`和`Purchase/Rental`
下面对`House/Apartment`和`Purchase/Rental`分别两两配对，注意新的子类既没有`__init__()`也没有`display()`方法，因为都在父类中实现了。默认的`__init__()`会调用父类的`__init__()`。

```python
class HouseRental(Rental, House):
    @classmethod
    def prompt_init(cls):
        init = House.prompt_init()
        init.update(Rental.prompt_init())
        return init


class ApartmentRental(Rental, Apartment):
    @classmethod
    def prompt_init(cls):
        init = Apartment.prompt_init()
        init.update(Rental.prompt_init())
        return init


class ApartmentPurchase(Purchase, Apartment):
    @classmethod
    def prompt_init(cls):
        init = Apartment.prompt_init()
        init.update(Purchase.prompt_init())
        return init


class HousePurchase(Purchase, House):
    @classmethod
    def prompt_init(cls):
        init = House.prompt_init()
        init.update(Purchase.prompt_init())
        return init
```

```python
init_args = HouseRental.prompt_init()
print('\n')
HouseRental(**init_args).display()
```

#### 设计`Agent`类
回顾一开始的需求：设计一个简单的房地产应用，允许`Agent`管理可出售或出租的`Property`（`Apartment`和`House`）。`Agent`可以添加`Property`，展示出当前空闲的`Porperty`。有了前面实现的各个对象（`Property`，`HouseRental`等），实现`Agent`就非常简单：

```python
class Agent:
    type_map = {
        ("house", "rental"): HouseRental,
        ("house", "purchase"): HousePurchase,
        ("apartment", "rental"): ApartmentRental,
        ("apartment", "purchase"): ApartmentPurchase
    }

    def __init__(self):
        self.property_list = []

    def display_properties(self):
        for property in self.property_list:
            property.display()

    def add_property(self):
        property_type = get_valid_input(
            "What type of property? ",
            ("house", "apartment")).lower()
        payment_type = get_valid_input(
            "What payment type? ",
            ("purchase", "rental")).lower()

        PropertyClass = self.type_map[(property_type, payment_type)]
        init_args = PropertyClass.prompt_init()
        self.property_list.append(PropertyClass(**init_args))
```

```python
agent = Agent()
agent.add_property()

print('\n')
agent.display_properties()
```

### 附录A：多重继承的`super().__init__()`
首先看下父类和子类都没有`super().__init__()`的情况：

```python
class Father1:
    def __init__(self):
        print('Father1')

class Father2:
    def __init__(self):
        print('Father2')

class Child1(Father1, Father2):
    def __init__(self):
        print('Child1')

Child1()
```

结果显示，`Child1`初始化时候只调用了自己的`__init__()`。

如果子类加入`super().__init__()`呢：

```python
class Child2(Father1, Father2):
    def __init__(self):
        super().__init__()
        print('Child2')

Child2()
```

结果显示，`Child2`初始化时候先调用了`Father1.__init__()`，然后再调用了自己的`__init__()`。

如果`Father1`加入`super().__init__()`呢：

```python
class Father1:
    def __init__(self):
        super().__init__()
        print('Father1')

class Child3(Father1, Father2):
    def __init__(self):
        super().__init__()
        print('Child3')

Child3()
```

结果显示，`Child3`初始化时候先调用了`Father1.__init__()`，`Father1`调用了`Father2.__init__()`，最后调用了自己的`__init__()`。

如果`Child`换个继承顺序呢：

```python
class Child4(Father2, Father1):
    def __init__(self):
        super().__init__()
        print('Child4')

Child4()
```

结果显示，`Child4`初始化时候先调用了`Father2.__init__()`，然后调用了自己的`__init__()`。

所以多重继承时候，父类的顺序和前一个父类是否有`super().__init__()`都会影响子类是否能正确调用多个父类的`__init__()`。另外由于多个父类参数不一样，所以在使用多重继承时候，每个父类都要加上`super().__init__(**kwargs)`，如下：

```python
class Father1:
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        print('Father1')

class Father2:
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        print('Father2')

class Child5(Father1, Father2):
    def __init__(self):
        super().__init__()
        print('Child5')

Child5()
```
