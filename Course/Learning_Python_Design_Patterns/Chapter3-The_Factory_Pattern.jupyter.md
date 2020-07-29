<!-- #region -->
# Chapter-3 工厂模式：建立创建对象的工厂
Factory模式有3种变体：
- 简单工厂模式：允许接口创建对象，但不会暴露对象的创建逻辑。
- 工厂方法模式：允许接口创建对象，但使用哪个类来创建对象，则是交由子类决定的。
- 抽象工厂模式：抽象工厂是一个能够创建一系列相关的对象而无需指定/公开其具体类的接口。该模式能够提供其他工厂的对象，在其内部创建其他对象。


## 简单工厂模式
这种模式下，工厂可以帮助开发人员**创建不同类型的对象，而不是直接将对象实例化**。下图是简单工厂的UML图，客户端类使用的是`Factory`类，该类具有`create_type()`方法。当客户端使用类型参数调用`create_type()`方法时，`Factory`会根据传入的参数，返回`Product1`或`Product2`。

![](http://static.zybuluo.com/AustinMxnet/z6do394regvm9mmc8768kw4i/image.png)

在下面的代码段中，我们将创建一个名为`Animal`的抽象产品。`Animal`是一个抽象的基类（`ABCMeta`是Python的特殊元类，用来生成类`Abstract`），它带有方法`do_say()`。我们利用`Animal`接口创建了两种产品（`Cat`和`Dog`），并实现了`do_say()`方法来提供这些动物的相应的叫声。`ForestFactory`是一个带有`make_sound()`方法的工厂。根据客户端传递的参数类型，它就可以在运行时创建适当的`Animal`实例，并输出正确的声音：
<!-- #endregion -->

```python
from abc import ABCMeta, abstractmethod

class Animal(metaclass=ABCMeta):
    @abstractmethod
    def do_say(self):
        pass

class Dog(Animal):
    def do_say(self):
        print("Bhow Bhow!!")

class Cat(Animal):
    def do_say(self):
        print("Meow Meow!!")

# forest factory defined
class ForestFactory(object):
    def make_sound(self, object_type):
        return eval(object_type)().do_say()
```

客户端的代码：

```python
ff = ForestFactory()
animal = 'Dog' # Dog or Cat
ff.make_sound(animal)
```

## 工厂方法模式
工厂方法模式的特点：
- 工厂定义了接口来创建对象，本身并不负责创建对象，而是将这一任务交由子类来完成，即子类决定了要实例化哪些类。
- 工厂方法的创建是**通过继承而不是通过实例化**来完成的。
- 工厂方法使设计更加具有可定制性。它可以返回相同的实例或子类，而不是某种类型的对象（就像在简单工厂方法中的那样）。

在下面的UML图中，有一个包含`factoryMethod()`方法的抽象类`Creator`。`FactoryMethod()`方法负责创建指定类型的对象。`ConcreteCreator`类提供了一个实现`Creator`抽象类的`factoryMethod()`方法，这种方法可以在运行时修改已创建的对象。`ConcreteCreator`创建`ConcreteProduct`，并确保其创建的对象实现了`Product`类，同时为`Product`接口中的所有方法提供相应的实现。

简而言之，`Creator`接口的`factoryMethod()`方法和`ConcreteCreator`类共同决定了要创建`Product`的哪个子类。因此，工厂方法模式定义了一个接口来创建对象，但具体实例化哪个类则是由它的子类决定的。

![](http://static.zybuluo.com/AustinMxnet/6gztekvrkf84rbz74c4zpi2n/image.png)

**e.g. 用一个现实世界的场景来理解工厂方法的实现：**  
假设我们想在不同类型的社交网络（例如LinkedIn、Facebook等）上为个人或公司建立简介。那么，每个简介都有某些特定的组成区：
- 在LinkedIn的简介中，有一个区是关于个人申请的专利或出版作品的。
- 在Facebook上，你将在相册中看到最近度假地点的照片区。
- 此外，在这两个简介中，都有一个个人信息的区。

因此，我们在创建不同类型的简介`Profile`（对应上图`Creator`）时，要添加正确的区`Section`（对应上图`Product`）到相应的简介中。首先定义接口`Section`抽象类，并提供抽象方法`describe()`，然后定义具体`ConcreteProduct`，也就是`PersonalSection`等：

```python
from abc import ABCMeta, abstractmethod

class Section(metaclass=ABCMeta):
    @abstractmethod
    def describe(self):
        pass

class PersonalSection(Section):
    def describe(self):
        print("Personal Section")

class AlbumSection(Section):
    def describe(self):
        print("Album Section")

class PatentSection(Section):
    def describe(self):
        print("Patent Section")

class PublicationSection(Section):
    def describe(self):
        print("Publication Section")
```

有了`Product`类，接着创建`Creator`抽象类，即`Profile`，`Profile`抽象类提供了一个工厂抽象方法`createProfile()`。这个抽象方法应该由`ConcreteClass`实现，去实际创建带有适当区`ConcreteProduct`的`Profile`。**由于`Profile`抽象类不知道简介应具有哪些区（例如Facebook的简介应该提供个人信息区和相册区），所以让子类`linkedin`等来决定这些事情**。

下面创建了两个`ConcreteCreator`类(`linkedin`和`facebook`)，每个类根据实际情况实现了`createProfile()`抽象方法，由该方法在运行时实际创建（实例化）多个区（`ConcreteProducts`）：

```python
class Profile(metaclass=ABCMeta):
    def __init__(self):
        self.sections = []
        self.createProfile()

    @abstractmethod
    def createProfile(self):
        pass

    def getSections(self):
        return self.sections

    def addSections(self, section):
        self.sections.append(section)


class linkedin(Profile):
    def createProfile(self):
        self.addSections(PersonalSection())
        self.addSections(PatentSection())
        self.addSections(PublicationSection())


class facebook(Profile):
    def createProfile(self):
        self.addSections(PersonalSection())
        self.addSections(AlbumSection())
```

最后客户端代码根据指定的选项创建所需的简介，即决定实例化哪个`Creator`类：

```python
profile_type = ['LinkedIn', 'FaceBook'][0]
profile = eval(profile_type.lower())()
print("Creating Profile..", type(profile).__name__)
print("Profile has sections --", profile.getSections())
```

通过示例可以看出工厂方法模式的优点：
1. 灵活性，使得代码通用，因为不必绑定到某个类进行实例化。这样，只依赖于接口（`Product`）而不依赖于`ConcreteProduct`类。
2. 松耦合性，因为创建对象的代码与使用对象的代码是分开的。客户端不必费心传递什么参数和实例化哪个类。而且增加新类非常容易，维护成本低。

## 抽象工厂模式
抽象工厂模式的主要目的是提供一个接口来创建一系列相关对象，而无需指定具体的类。工厂方法将创建实例的任务委托给了子类，而抽象工厂方法的目标是创建一系列相关对象。如下图所示，`ConcreteFactory1`和`ConcreteFactory2`是通过`AbstractFactory`接口创建的。此接口具有创建多种产品的相应方法。

![](http://static.zybuluo.com/AustinMxnet/a27h38w9pswewt4iwnl6t1ii/image.png)

`ConcreteFactory1`和`ConcreteFactory2`实现了`AbstractFactory`，并分别创建实例`ConcreteProduct1`、`ConcreteProduct2`和`AnotherConcreteProduct1`、`AnotherConcreteProduct2`。

实际上，抽象工厂模式不仅确保**客户端与对象的创建相互隔离**，同时还确保客户端能够使用创建的对象。但是，客户端只能通过接口访问对象，如果要使用一个系列中的多个产品，那么抽象工厂模式能够帮助客户端一次使用来自一个产品/系列的多个对象。例如，如果正在开发的应用应该与平台无关，它需要对各种依赖项进行抽象处理，这些依赖项包括操作系统、文件系统调用等等。抽象工厂模式负责为整个平台创建所需的服务，这样的话，客户端就不必直接创建平台对象了。

**e.g. 实际案例：披萨店**  
假设我们开办了一家披萨店，供应美味的印式和美式披萨饼。我们首先创建一个抽象基类`PizzaFactory`（`AbstractFactory`见前面的UML图）。`PizzaFactory`类有两个抽象方法即`createVegPizza()`和`createNonVegPizza()`，它们需要通过`ConcreteFactory`实现。在这个例子中，我们创造了两个具体的工厂，分别名为`IndianPizzaFactory`和`USPizzaFactory`：

```python
from abc import ABCMeta, abstractmethod

class PizzaFactory(metaclass=ABCMeta):
    @abstractmethod
    def createVegPizza(self):
        pass

    @abstractmethod
    def createNonVegPizza(self):
        pass


class IndianPizzaFactory(PizzaFactory):
    def createVegPizza(self):
        return DeluxVeggiePizza()

    def createNonVegPizza(self):
        return ChickenPizza()


class USPizzaFactory(PizzaFactory):
    def createVegPizza(self):
        return MexicanVegPizza()

    def createNonVegPizza(self):
        return HamPizza()
```

现在进一步定义`AbstractProducts`，将创建两个抽象类：`VegPizza`和`NonVegPizza`。它们都定义了自己的方法，分别是`prepare()`和`serve()`。然后为每个`AbstractProducts`定义`ConcreteProducts`，就本例而言，`ConcreteProducts1`和`ConcreteProducts2`分别对应`DeluxVeggiePizza`和`MexicanVegPizza`，`AnotherConcreteProducts1`和`AnotherConcreteProducts2`分别对应`ChickenPizza`和`HamPizza`。

```python
class VegPizza(metaclass=ABCMeta):
    @abstractmethod
    def prepare(self, VegPizza):
        pass

class NonVegPizza(metaclass=ABCMeta):
    @abstractmethod
    def serve(self, VegPizza):
        pass

class DeluxVeggiePizza(VegPizza):
    def prepare(self):
        print("Prepare ", type(self). __name__)

class ChickenPizza(NonVegPizza):
    def serve(self, VegPizza):
        print(type(self). __name__, " is served with Chicken on ",
              type(VegPizza). __name__)

class MexicanVegPizza(VegPizza):
    def prepare(self):
        print("Prepare ", type(self). __name__)

class HamPizza(NonVegPizza):
    def serve(self, VegPizza):
        print(type(self). __name__, " is served with Ham on ",
              type(VegPizza). __name__)
```

```python
class PizzaStore:
    def __init__(self):
        pass
    def makePizzas(self):
        for factory in [IndianPizzaFactory(), USPizzaFactory()]:
            self.factory = factory
            self.NonVegPizza = self.factory.createNonVegPizza()
            self.VegPizza = self.factory.createVegPizza()
            self.VegPizza.prepare()
            self.NonVegPizza.serve(self.VegPizza)

pizza = PizzaStore()
pizza.makePizzas()
```

## 工厂模式与抽象工厂模式

|   工厂模式   | 抽象工厂模式 |
|  :--:  | :--:  |
| 向客户端开放了一个创建对象的方法   | 包含一个或多个工厂方法来创建一个系列的相关对象 |
| 使用继承和子类来决定要创建何种对象  | 使用组合委派责任创建其他类的对象 |
| 创建一个产品                |  创建相关产品的系列|
