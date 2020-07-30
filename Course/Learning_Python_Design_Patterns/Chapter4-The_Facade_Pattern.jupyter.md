# Chapter-4 门面模式 —— 与门面相适
The Façade Pattern - Being Adaptive with Façade

工厂模式和单例模式都属于**创建型设计模式**，这一章介绍的门面模式属于**结构型设计模式**。

## 结构性设计模式
以下几点将有助于我们更好地了解结构型设计模式。
- 结构型模式描述如何将对象和类组合成更大的结构。
- 结构型模式是一种能够简化设计工作的模式，因为它能够找出更简单的方法来认识或表示实体之间的关系（面向对象中，实体指的是对象或类）。
- 类Class通过继承来描述抽象，从而提供更有用的程序接口；而对象Object则描述了如何将对象联系起来从而组合成更大的对象。**结构型模式是类和对象模式的综合体**。

下面给出结构型设计模式的几个例子，它们都是通过对象或类之间的交互来实现更高级的设计或架构目标：
- **适配器模式**：将一个接口转换成客户希望的另外一个接口。它试图根据客户端的需求来匹配不同类的接口。
- **桥接模式**：该模式将对象的接口与其实现进行解耦，使得两者可以独立工作。
- **装饰器模式**：该模式允许在运行时或以动态方式为对象添加职责。我们可以通过接口给对象添加某些属性。

## 门面模式
门面（facade）通常是指建筑物的表面，尤其是最有吸引力的那一面。它也可以表示一种容易让人误解某人的真实感受或情况的行为或面貌。当人们从建筑物外面经过时，可以欣赏其外部面貌，却不了解建筑物结构的复杂性。这就是门面模式的使用方式。门面在隐藏内部系统复杂性的同时，为客户端提供了一个接口，以便它们可以非常轻松地访问系统。

假设你要到某个商店去买东西，但是你对这个商店的布局并不清楚。通常情况下，你会去找店主，只要你告诉他/她要买什么，店主就会把这些商品拿给你。顾客不必了解店面的情况，可以通过一个简单的接口来完成购物，这里的接口就是店主。

门面设计模式实际上完成了下列事项：
- 它为子系统中的一组接口提供一个统一的接口，并定义一个高级接口来帮助客户端通过更加简单的方式使用子系统。
- 门面所解决问题是，如何用单个接口对象来表示复杂的子系统。**实际上，它并不是封装子系统，而是对底层子系统进行组合**。
- 它利于解耦多个客户端的实现。

门面模式的UML类图如下：

![](http://static.zybuluo.com/AustinMxnet/fck6dum3f6dk10pzzcqxgkb7/image.png)

如图所示，门面模式有3个主要的参与者：
- 门面：门面的主要责任是，将一组复杂导致系统封装起来，从而为外部世界提供一个舒适的外观。
  - 它是一个接口，它知道某个请求可以交由哪个子系统进行处理。
  - 它使用组合将客户端的请求委派给相应的子系统对象。

- 系统：这代表一组不同的子系统，使整个系统混杂在一起，难以观察或使用。
  - 它实现子系统的功能，同时，系统由一个类表示。理想情况下，系统应该由一组负责不同任务的类来表示。
  - 它处理门面对象分配的工作，但并不知道门面，而且不引用它。

- 客户端：客户端与门面进行交互，这样就可以轻松地与子系统进行通信并完成工作了。不必担心系统的复杂性。
  - 客户端是实例化门面的类。
  - 为了让子系统完成相应的工作，客户端需要向门面提出请求。
 
## e.g. 婚礼安排
假设你要举行一场婚礼，这是一个艰巨的任务，需要预订一家酒店或场地、与餐饮人员交代酒菜、布置场景、并安排背景音乐等。你可以自己搞定一切，例如找相关人员谈话、与他们进行协调、敲定价格等。此外，你还可以去找会务经理，让他/她为你处理这些事情。会务经理负责跟各个服务提供商交涉，并为你争取最优惠的价格。

我们从门面模式的角度来看待这些事情：
- 客户端：你需要在婚礼前及时完成所有的准备工作。每一项安排都应该是顶级的，这样客人才会喜欢这些庆祝活动。
- 门面：会务经理负责与所有相关人员进行交涉，这些人员负责处理食物、花卉装饰等。
- 子系统：它们代表提供餐饮、酒店管理和花卉装饰等服务的系统。

`EventManager`扮演了门面的角色，简化了你的工作。门面负责与子系统进行交流，并代表你为婚姻完成所有的预订和准备工作：

```python
class EventManager(object):
    def __init__(self):
        print("Event Manager:: Let me talk to the folks\n")

    def arrange(self):
        self.hotelier = Hotelier()
        self.hotelier.bookHotel()

        self.florist = Florist()
        self.florist.setFlowerRequirements()

        self.caterer = Caterer()
        self.caterer.setCuisine()

        self.musician = Musician()
        self.musician.setMusicType()
```

下面是各个子系统：

```python
class Hotelier(object):
    def __init__(self):
        print("Arranging the Hotel for Marriage? --")

    def __isAvailable(self):
        print("Is the Hotel free for the event on given day?")
        return True

    def bookHotel(self):
        if self.__isAvailable():
            print("Registered the Booking\n\n")


class Florist(object):
    def __init__(self):
        print("Flower Decorations for the Event? --")

    def setFlowerRequirements(self):
        print("Carnations, Roses and Lilies would be used for Decorations\n\n")


class Caterer(object):
    def __init__(self):
        print("Food Arrangements for the Event --")

    def setCuisine(self):
        print("Chinese & Continental Cuisine to be served\n\n")


class Musician(object):
    def __init__(self):
        print("Musical Arrangements for the Marriage --")

    def setMusicType(self):
        print("Jazz and Classical will be played\n\n")
```

你很聪明，将这些事情都委托给了会务经理。在`You`中，创建了一个`EventManager`类的对象，这样经理就会通过与相关人员进行交涉来筹备婚礼，而你则可以找个地方喝大茶了：

```python
class You(object):
    def __init__(self):
        print("You:: Whoa! Marriage Arrangements??!!!")

    def askEventManager(self):
        print("You:: Let's Contact the Event Manager\n\n")
        em = EventManager()
        em.arrange()

    def __del__(self):
        print("You:: Thanks to Event Manager, all preparations done! Phew!")


you = You()
you.askEventManager()
del you
```
