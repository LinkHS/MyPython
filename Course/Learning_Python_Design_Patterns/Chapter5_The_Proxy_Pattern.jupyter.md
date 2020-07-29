# Chapter-5 代理模式——控制对象的访问
代理模式和门面模式一样，也属于**结构性设计模式**。

代理通常就是一个介于寻求方和提供方之间的中介系统。寻求方是发出请求的一方，而提供方则是根据请求提供资源的一方。在Web世界中，它相当于代理服务器。客户端（万维网中的用户）在向网站发出请求时，首先连接到代理服务器，向它请求诸如网页之类的资源。代理服务器在内部评估此请求，将其发送到适当的服务器，当它收到响应后，就会将响应传递给客户端。因此，代理服务器可以封装请求、保护隐私，并且非常适合在分布式架构中运行。

在设计模式的上下文中，代理是充当实际对象接口的类。对象类型可以是多样化的，例如网络连接、内存和文件中的大对象，等等。简而言之，代理就是封装实际服务对象的包装器或代理人。代理可以为其包装的对象提供附加功能，而无需更改对象的代码。代理模式的主要目的是为其他对象提供一个代理者或占位符，从而控制对实际对象的访问。

代理模式可以用于多种场景:
- **以更简单的方式表示一个复杂的系统**。例如，涉及多个复杂计算或过程的系统应该提供一个更简单的接口，让它充当客户端的代理。
- **提高现有的实际对象的安全性**。在许多情况下，都不允许客户端直接访问实际对象。这是因为实际对象可能受到恶意活动的危害。这时候，代理就能起到抵御恶意活动的盾牌作用，从而保护了实际的对象。
- **为不同服务器上的远程对象提供本地接口**。一个明显的例子是客户端希望在远程系统上运行某些命令的分布式系统，但客户端可能没有直接的权限来实现这一点。因此它将请求转交给本地对象（代理），然后由远程机器上的代理执行该请求。
- **为消耗大量内存的对象提供了一个轻量级的句柄**。一个典型的例子是网站用户的个人简介头像，最好在列表视图中显示简介头像的缩略图，而在需要展示用户详细介绍时，再加载实际图片。

不妨以演员与他的经纪人为例，当制作公司想要找演员拍电影时，他们通常会与经纪人交流，而不是直接跟演员交流。经纪人会根据演员的日程安排和其他合约情况，来答复制作公司该演员是否有空，以及是否对该影片感兴趣和片酬问题等。

下面的Python代码实现了这种场景，代理`Agent`用于查看`Actor`是否正处于忙碌状态：

```python
class Actor(object):
    def __init__(self):
        self.isBusy = False

    def occupied(self):
        self.isBusy = True
        print(type(self).__name__, "is occupied with current movie")

    def available(self):
        self.isBusy = False
        print(type(self).__name__, "is free for the movie")

    def getStatus(self):
        return self.isBusy


class Agent(object):
    def __init__(self):
        self.principal = None

    def work(self):
        self.actor = Actor()
        if self.actor.getStatus():
            self.actor.occupied()
        else:
            self.actor.available()


r = Agent()
r.work()
```

![](http://static.zybuluo.com/AustinMxnet/6d7bfm3nx1ta5u6e4rig9l2p/image.png)

## 不同类型的代理
### 虚拟代理
如果一个对象实例化后会占用大量内存的话，可以先利用占位符来表示，这就是虚拟代理。例如，假设你想在网站上加载大型图片，而这个请求需要很长时间才能加载完成。通常，开发人员将在网页上创建一个占位符图标，以提示这里有图像。但是，只有当用户实际点击图标时才会加载图像，从而节省了向存储器中加载大型图像的开销。因此，在虚拟代理中，当客户端请求或访问对象时，才会创建实际对象。

### 远程代理
它给位于远程服务器或不同地址空间上的实际对象提供了一个本地表示。例如，你希望为应用程序建立一个监控系统，而该应用涉及多个Web服务器、数据库服务器、芹菜（celery）任务服务器、缓存服务器，等等。如果我们要监视这些服务器的CPU和磁盘利用率，就需要建立一个对象，该对象能够用于监视应用程序运行的上下文中，同时还可以执行远程命令以获取实际的参数值。

### 保护代理
这种代理能够控制`RealSubject`的敏感对象的访问。例如分布式系统中，Web应用会提供多个服务，这些服务相互协作来提供各种功能，如认证服务充当负责认证和授权的保护性代理服务器。在这种情况下，代理自然有助于保护网站的核心功能，防止无法识别或未授权的代理访问它们。

### 智能代理
智能代理在访问对象时插入其他操作。例如，系统中有一个核心组件，它将状态信息集中保存在一个地点。通常情况下，这样的组件需要被多个不同的服务调用，可能导致共享资源的问题。智能代理是内置的，在访问核心组件之前会检查实际对象是否被锁定。

## e.g. 刷卡消费
假设你在商场溜达，看中了一件漂亮的牛仔衫。你想买这件衬衫，但手里的现金却不够了。这时你可以去银行取钱，再回来付款。由于银行在商家处有刷卡机（代理），所以你只要在商家刷一下借记卡，这笔钱就会划入商家的账户，从而完成支付`do_pay()`：

```python
from abc import ABCMeta, abstractmethod

class Payment(metaclass=ABCMeta):
    @abstractmethod
    def do_pay(self):
        pass
```

`Bank`除了实现`do_pay()`，还有别的操作，这里主要是检查账户有效性、余额等：

```python
class Bank(Payment):
    def __init__(self):
        self.card = None
        self.account = None

    def __getAccount(self):
        # Assume card number is account number
        self.account = self.card
        return self.account

    def __hasFunds(self):
        print("Bank:: Checking if Account",
              self.__getAccount(), "has enough funds")
        return True

    def setCard(self, card):
        self.card = card

    def do_pay(self):
        if self.__hasFunds():
            print("Bank:: Paying the merchant")
            return True
        else:
            print("Bank:: Sorry, not enough funds!")
            return False
```

代理需要获取客户的银行卡号，并发送给银行：

```python
class DebitCard(Payment):
    def __init__(self):
        self.bank = Bank()

    def do_pay(self):
        #card = input("Proxy:: Punch in Card Number: ")
        print("Proxy:: Punch in Card Number"); card = "62220120"
        self.bank.setCard(card)
        return self.bank.do_pay()
```

```python
class You:
    def __init__(self):
        print("You:: Lets buy the Denim shirt")
        self.debitCard = DebitCard()
        self.isPurchased = None

    def make_payment(self):
        self.isPurchased = self.debitCard.do_pay()

    def __del__(self):
        if self.isPurchased:
            print("You:: Wow! Denim shirt is Mine :-)")
        else:
            print("You:: I should earn more :(")

you = You()
you.make_payment()
del you
```

## 门面模式和代理模式的比较

| 代理模式 | 门面模式 |
| :------: | :------: |
| 为其他对象提供了代理或占位符，以控制对原始对象的访问 | 为类的大型子系统提供了一个接口 |
| 代理对象具有与其目标对象相同的接口，并保存有目标对象的引用 | 实现了子系统之间的通信和依赖性的最小化 |
