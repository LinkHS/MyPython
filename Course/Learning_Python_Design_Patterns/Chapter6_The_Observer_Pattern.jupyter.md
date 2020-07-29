<!-- #region -->
# Chapter-6 观察者模式——了解对象的情况
## 行为设计模式
之前介绍了创建型设计模式和结构性设计模式，这章介绍第三种**行为设计模式**。

**创建型模式**的工作原理是基于对象的创建机制的。它隔离了对象的创建细节，创建代码不依赖创建对象的类型。**结构型模式**用于设计对象和类的结构，使它们可以相互协作以形成更大的结构，重点关注的是简化结构以及识别类和对象之间的关系。

**行为型模式**，顾名思义，主要关注对象的责任，处理对象之间的交互，以实现更大的功能。行为型模式建议：对象之间应该能够彼此交互，同时还应该是松散耦合的。


## 观察者模式
观察者模式是最简单的行为型模式之一。

在观察者设计模式中，对象（主题）维护了一个依赖（观察者）列表，以便主题可以使用观察者定义的任何方法通知所有观察者它所发生的变化。

例如分布式应用中，多个服务通常通过彼此交互来实现用户想要实现的更大型操作。服务可以执行多种操作，但是直接或很大程度上取决于与其交互的服务对象的状态。其他还有广播或发布/订阅系统、股票市场等。

因此，如果应用中存在一个**许多其他服务所依赖的核心服务**，那么该核心服务就会成为观察者必须观察/监视其变化的主题。当主题发生变化时，观察者应该改变自己的对象的状态，或者采取某些动作。这种情况（其中从属服务监视核心服务的状态变化）描述了观察者设计模式的经典情景。观察者模式的主要目标如下：
- 定义了对象之间的一对多的依赖关系，从而使得一个对象中的任何更改都将自动通知给其他依赖对象；
- 封装了主题的核心组件；

下面是一个典型的观察者实现，观察者通过`register`和主体联系起来，主体通过`notify`调用所有观察者：
<!-- #endregion -->

```python
class Subject:
    def __init__(self):
        self.__observers = []

    def register(self, observer):
        self.__observers.append(observer)

    def notifyAll(self, *args, **kwargs):
        for observer in self.__observers:
            observer.notify(self, *args, **kwargs) 

class Observer1:
    def __init__(self, subject):
        subject.register(self)

    def notify(self, subject, *args):
        print(type(self). __name__,':: Got', args, 'From', subject)

class Observer2:
    def __init__(self, subject):
        subject.register(self)

    def notify(self, subject, *args):
        print(type(self). __name__, ':: Got', args, 'From', subject)

subject = Subject()
observer1 = Observer1(subject)
observer2 = Observer2(subject)
subject.notifyAll('notification')
```

![](http://static.zybuluo.com/AustinMxnet/4hvywuweirpxf7203j7qnn1s/image.png)

## e.g. 新闻订阅
我们以新闻机构为例来展示观察者模式。新闻机构通常从不同地点收集新闻，并将其发布给订阅者。由于信息是实时发送或接收的，所以新闻机构应该尽快向其订户公布该消息。此外，随着技术的进步，订户不仅可以订阅报纸，而且可以通过其他的形式进行订阅，例如电子邮件、移动设备、短信或语音呼叫。所以，还应该具备在将来添加任意其他订阅形式的能力，以便为未来的新技术做好准备。

先看下观察者抽象类`Subscriber`，需要提供`update()`共新闻机构发布消息：

```python
from abc import ABCMeta, abstractmethod

class Subscriber(metaclass=ABCMeta):
    @abstractmethod
    def update(self):
        pass
```

本例中给用户提供短信、邮件和其他三种订阅方式，**由于Python动态语言的鸭子类型特点，这里并不需要继承`Subscriber`，只需要提供对应的接口`update()`即可**：

```python
class SMSSubscriber:
    def __init__(self, publisher):
        self.publisher = publisher
        self.publisher.attach(self)

    def update(self):
        print(type(self).__name__, self.publisher.getNews())


class EmailSubscriber:
    def __init__(self, publisher):
        self.publisher = publisher
        self.publisher.attach(self)

    def update(self):
        print(type(self).__name__, self.publisher.getNews())


class AnyOtherSubscriber:
    def __init__(self, publisher):
        self.publisher = publisher
        self.publisher.attach(self)

    def update(self):
        print(type(self).__name__, self.publisher.getNews())
```

可以看到各个`Subscriber`保存了`publisher`的对象，所以通知时候在各个`Subscriber`内部调用了`publisher.getNews()`，也可以不保存`publisher`对象，`publisher`通过`update`通知到`Subscriber`，由用户决定是否查看，请看下一章节通知方式中的**推模型和拉模型**。

最后实现新闻机构类：

```python
class NewsPublisher:
    def __init__(self):
        self.__subscribers = []
        self.__latestNews = None

    def attach(self, subscriber):
        self.__subscribers.append(subscriber)

    def detach(self):
        return self.__subscribers.pop()

    def subscribers(self):
        return [type(x).__name__ for x in self.__subscribers]

    def notifySubscribers(self):
        for sub in self.__subscribers:
            sub.update()

    def addNews(self, news):
        self.__latestNews = news

    def getNews(self):
        return "Got News:", self.__latestNews
```

测试过程，分别初始化`SMSSubscriber`、`EmailSubscriber`和`AnyOtherSubscriber`类得到3个观察者，然后注册，发送通知：

```python
news_publisher = NewsPublisher()
for Subscribers in [SMSSubscriber, EmailSubscriber, AnyOtherSubscriber]:
    Subscribers(news_publisher)
print("\nSubscribers:", news_publisher.subscribers())

news_publisher.addNews('Hello World!')
news_publisher.notifySubscribers()

print("\nDetached:", type(news_publisher.detach()).__name__)
print("\nSubscribers:", news_publisher.subscribers())

news_publisher.addNews('My second news!')
news_publisher.notifySubscribers()
```

## 通知方式
有两种不同的方式可以通知观察者：推模型或拉模型。

### 拉模型
在拉模型中，观察者扮演积极的角色：
- 每当发生变化时，主题都会向所有已注册的观察者进行广播。
- 出现变化时，观察者负责获取相应的变化情况，或者从订户那里拉取数据。

**拉模型的效率较低**，因为它涉及两个步骤，第一步，通知观察者；第二步，观察者从主体那里提取所需的数据。

### 推模型
在推模型中，主体是起主导作用的一方，不仅通知观察者，而且还向观察者发送详细的信息（**即使不需要**）。由于只从主体发送所需的数据，能提高效率，但是当发送大量观察者用不到的数据时，会使响应时间过长。

所以当数据量很大的时候，要根据不同语言或者实现方式（例如线程、进程、Python的协程）特点优化，例如共享内存、通知优先级等方式。
