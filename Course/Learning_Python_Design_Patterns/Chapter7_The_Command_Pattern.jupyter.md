# Chapter-7 命令模式——封装调用
**行为模式侧重于对象的响应性**，它利用对象之间的交互实现更强大的功能。**命令模式也是一种行为设计模式**，其中对象用于封装在完成一项操作时或在触发一个事件时所需的全部信息。这些信息包括以下内容：
- 方法名称
- 拥有方法的对象
- 方法参数的值

例如PC软件安装时，安装向导通过多个步骤或窗口来得到用户的偏好设置。通常来说，向导可以使用命令模式来实现。首先启动一个名为`Command`的对象，用户在向导的多个步骤中指定的选项将存储在这个对象中。当用户在向导的最后一步单击Finish按钮时，`Command`对象就会运行`execute()`方法，该方法会根据所有存储的选项并完成相应的安装过程。因此，关于选择的**所有信息被封装在稍后用于采取动作的对象中**。

另一个简单的例子是打印机后台处理程序。它可以用`Command`对象的形式来实现，该对象用于存储页面类型（A5-A1）、纵向/横向、分选/不分选等信息。当用户打印东西（例如图像）时，假脱机程序就会运行Command对象的execute()方法，并使用设置的首选项打印图像。

在下面的示例中，我们首先在客户端代码中创建`Wizard`对象，然后使用`preferences()`方法存储用户在向导的各个屏幕期间做出的选择。在向导中单击Finish按钮时，就会调用`execute()`方法，`execute()`方法将会根据首选项来开始安装：

```python
class Wizard():
    def __init__(self, src, rootdir):
        self.choices = []
        self.rootdir = rootdir
        self.src = src

    def preferences(self, command):
        self.choices.append(command)

    def execute(self):
        for choice in self.choices:
            if list(choice.values())[0]:
                print(choice.keys(), "Copying binaries --", self.src, " to ", self.rootdir)
            else:
                print(choice.keys(), "No Operation")


# Client code
wizard = Wizard('python3.5.gzip', '/usr/bin/')
# Users chooses to install Python only
wizard.preferences({'python': True})
wizard.preferences({'java': False})
wizard.execute()
```

![](http://static.zybuluo.com/AustinMxnet/2rjkq9g9elxubcj67a1vhgoq/image.png)

通过该UML图不难发现，该模式主要涉及5个参与者：
- `Command`：声明执行操作的接口
- `ConcreteCommand`：将一个`Receiver`对象和一个操作绑定在一起
- `Client`：创建`ConcreteCommand`对象并设定其接收者
- `Invoker`：要求该`ConcreteCommand`执行这个请求
- `Receiver`：知道如何实施与执行一个请求相关的操作

```python
from abc import ABCMeta, abstractmethod


class Command(metaclass=ABCMeta):
    def __init__(self, recv):
        self.recv = recv

    @abstractmethod
    def execute(self):
        pass


class ConcreteCommand(Command):
    def __init__(self, recv):
        self.recv = recv

    def execute(self):
        self.recv.action()


class Receiver:
    def action(self):
        print("Receiver Action")


class Invoker:
    def command(self, cmd):
        self.cmd = cmd

    def execute(self):
        self.cmd.execute()


recv = Receiver()
cmd = ConcreteCommand(recv)
invoker = Invoker()
invoker.command(cmd)
invoker.execute()
```

## e.g. 证券交易
我们通过一个证券交易所的例子来演示命令模式的实现。在证券交易所会发生哪些事情呢？作为用户，你会创建买入或卖出股票的订单。通常情况下，你无法直接执行买入或卖出，而是通过代理或经纪人负责将你的请求提交给证券交易所，完成工作。假设星期日晚上，你想在第二天早上开市后卖出股票，虽然交易所尚未开市，你仍然可以向代理提出卖出股票的请求。代理会将该请求放入排队，以便在星期一早晨当交易所开市的时候执行该请求，完成相应的交易。这实际上就是一个命令模式的经典情形。

**设计注意事项**
通过UML图可以看到，命令模式有4个主要参与者——`Command`、`ConcreteCommand`、`Invoker`和`Receiver`。对于前面的案例来说，我们应该创建一个`Order`接口，来定义客户端下达的订单。我们还应该定义`ConcreteCommand`类来买卖股票。此外还需为证券交易所定义一个类`StockTrade`，实际执行交易的`Receiver`类，以及接收订单并交由接收者执行的代理（称为调用者）。


```python
from abc import ABCMeta, abstractmethod

class Order(metaclass=ABCMeta):
    @abstractmethod
    def execute(self):
        pass
```

```python
class BuyStockOrder(Order):
    def __init__(self, stock):
        self.stock = stock

    def execute(self):
        self.stock.buy()


class SellStockOrder(Order):
    def __init__(self, stock):
        self.stock = stock

    def execute(self):
        self.stock.sell()
```

```python
class StockTrade:
    def buy(self):
        print("You will buy stocks")

    def sell(self):
        print("You will sell stocks")
```

```python
class Agent:
    def __init__(self):
        self.__orderQueue = []

    def placeOrder(self, order):
        self.__orderQueue.append(order)
        order.execute()
```

```python
#Client
stock = StockTrade()
buyStock = BuyStockOrder(stock)
sellStock = SellStockOrder(stock)

#Invoker
agent = Agent()
agent.placeOrder(buyStock)
agent.placeOrder(sellStock)
```

<!-- #region -->
## 其他命令模式
在软件中应用命令模式的方式有很多种。我们将讨论与云应用密切相关的两个实现：

- 重做或回滚操作： 
  - 在实现回滚或重做操作时，开发人员可以做两件不同的事情；
  - 这些是在文件系统或内存中创建快照，当被要求回滚时，恢复到该快照；
  - 使用命令模式时，


- 异步任务执行：
  - 在分布式系统中，我们通常要求设备具备异步执行任务的功能，以便核心服务在大量请求涌来时不会发生阻塞。
  - 在命令模式中，Invoker对象可以维护一个请求队列，并将这些任务发送到Receiver对象，以便它们可以独立于主应用程序线程来完成相应的操作。
<!-- #endregion -->
