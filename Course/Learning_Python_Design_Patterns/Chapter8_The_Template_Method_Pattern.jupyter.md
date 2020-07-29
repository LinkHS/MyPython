# Chapter-8 模板方法模式——封装算法
之前已经讨论过**行为模式**主要关注对象的响应性，它处理对象之间的交互以实现更强大的功能。**模板方法模式**也是一种行为设计模式，通过模板的方式来定义程序框架或算法，将这些步骤中的**一些实现推迟到子类**重新定义或定制算法的某些步骤。

按照软件开发术语来说，我们可以使用**抽象类**来定义算法的步骤，这些步骤在模板方法模式的上下文中也称为原始操作。抽象方法定义步骤，算法模板定义算法，`ConcreteClass`（子类化抽象类）则用来实现子类算法中的特定步骤。模板方法模式适用于以下场景：
- 当多个算法或类实现类似或相同逻辑的时候
- 在子类中实现算法有助于减少重复代码的时候
- 可以让子类利用覆盖实现行为来定义多个算法的时候

一个简单的例子是编译器。编译器本质上做两件事：收集源代码并将其编译为目标对象。如果需要为iOS设备定义交叉编译器，我们可以用模板方法模式实现：

```python
from abc import ABCMeta, abstractmethod


class Compiler(metaclass=ABCMeta):
    @abstractmethod
    def collectSource(self):
        pass

    @abstractmethod
    def compileToObject(self):
        pass

    @abstractmethod
    def run(self):
        pass

    def compileAndRun(self):
        self.collectSource()
        self.compileToObject()
        self.run()


class iOSCompiler(Compiler):
    def collectSource(self):
        print("Collecting Swift Source Code")

    def compileToObject(self):
        print("Compiling Swift code to LLVM bitcode")

    def run(self):
        print("Program runing on runtime environment")


iOS = iOSCompiler()
iOS.compileAndRun()
```

可以看出模板方法模式的主要意图如下：
- 使用基本操作定义算法的框架
- 重新定义子类的某些操作，而无需修改算法的结构
- 实现代码重用并避免重复工作
- 利用好通用接口或实现

观察下面的UML类图：
- `AbstractClass`：声明一个定义算法步骤的接口。在抽象方法的帮助下定义算法的操作或步骤，这些步骤将被具体子类覆盖。
- `ConcreteClass`：定义子类特定的步骤。实现（由抽象方法定义的）步骤，来执行算法子类的特定步骤。
- `template_method()`：定义算法的框架。在模板方法中调用抽象方法定义的多个步骤来定义序列或算法本身（即上例中的`compileAndRun`）。

![](http://static.zybuluo.com/AustinMxnet/dijbpvq3xpghg9wxoussgxt2/image.png)


## e.g. 旅行社
例如旅行社通常是如何运作的呢？ 他们定义了各种旅游路线，并提供度假套装行程。旅行涉及一些详细信息，如游览的地点、交通方式和与旅行有关的其他因素。当然，**同样的行程可以根据客户的需求进行不同的定制**。这种情况下，模板方法模式就有了用武之地。

抽象对象由`Trip`类表示，它是一个接口（抽象基类），定义了交通方式、参观地点、每天游玩等抽象方法：

```python
from abc import abstractmethod, ABCMeta


class Trip(metaclass=ABCMeta):
    @abstractmethod
    def setTransport(self):
        pass

    @abstractmethod
    def day1(self):
        pass

    @abstractmethod
    def day2(self):
        pass

    @abstractmethod
    def day3(self):
        pass

    @abstractmethod
    def returnHome(self):
        pass

    def itinerary(self):
        self.setTransport()
        self.day1()
        self.day2()
        self.day3()
        self.returnHome()
```

威尼斯Venice和马尔代夫Maldives两个行程实现各自不同的细节：

```python
class VeniceTrip(Trip):
    def setTransport(self):
        print("Take a boat and find your way in the Grand Canal")

    def day1(self):
        print("Visit St Mark's Basilica in St Mark's Square")

    def day2(self):
        print("Appreciate Doge's Palace")

    def day3(self):
        print("Enjoy the food near the Rialto Bridge")

    def returnHome(self):
        print("Get souvenirs for friends and get back")
```

```python
class MaldivesTrip(Trip):
    def setTransport(self):
        print("On foot, on any island, Wow!")

    def day1(self):
        print("Enjoy the marine life of Banana Reef")
    
    def day2(self):
        print("Go for the water sports and snorkelling")

    def day3(self):
        print("Relax on the beach and enjoy the sun")

    def returnHome(self):
        print("Dont feel like leaving the beach..")
```

```python
class TravelAgency:
    def arrange_trip(self, choice):
        print("What kind of place you'd like to go historical or to a beach?: %s" % choice)
        if choice == 'historical':
            self.trip = VeniceTrip()
        elif choice == 'beach':
            self.trip = MaldivesTrip()
        else:
            return
        self.trip.itinerary()


TravelAgency().arrange_trip('historical')
```

```python
TravelAgency().arrange_trip('beach')
```
