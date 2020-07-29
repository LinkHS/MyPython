# Chapter10 状态设计模式
状态模式也属于行为模式的范畴。在此模式中，一个对象可以基于其内部状态封装多个行为。状态模式也可以看作是在运行时改变对象行为的一种方式。**实际上，在运行时改变行为正好是Python所擅长的事情！**

状态设计模式在3个主要参与者的协助下工作：
- State：封装对象行为的接口，对象行为与对象的状态相关联。
- ConcreteState：实现State接口的子类，也就是与对象的特定状态相关联的实际行为。
- Context：定义了客户感兴趣的接口。Context还**维护一个ConcreteState子类的实例**，该子类在内部定义了对象的特定状态的实现。

![](http://static.zybuluo.com/AustinMxnet/cqt7m7w7njixo00052pruqqx/image.png)

```python
from abc import abstractmethod, ABCMeta


class State(metaclass=ABCMeta):
    @abstractmethod
    def Handle(self):
        pass


class ConcreteStateB(State):
    def Handle(self):
        print("ConcreteStateB")


class ConcreteStateA(State):
    def Handle(self):
        print("ConcreteStateA")


class Context(State):
    def __init__(self):
        self.state = None

    def getState(self):
        return self.state

    def setState(self, state):
        self.state = state

    def Handle(self):
        self.state.Handle()


context = Context()
stateA = ConcreteStateA()
stateB = ConcreteStateB()
context.setState(stateA)
context.Handle()
```

例如用按钮控制电视遥控器，`State`接口将会定义相应的方法（`doThis()`）来执行诸如打开/关闭电视等操作。我们还需要定义`ConcreteState`类来处理不同的状态：

```python
from abc import abstractmethod, ABCMeta


class State(metaclass=ABCMeta):
    @abstractmethod
    def doThis(self):
        pass


class StartState(State):
    def doThis(self):
        print("TV Switching ON..")


class StopState(State):
    def doThis(self):
        print("TV Switching OFF..")


class TVContext(State):
    def __init__(self):
        self.state = None

    def getState(self):
        return self.state

    def setState(self, state):
        self.state = state

    def doThis(self):
        self.state.doThis()


context = TVContext()
context.getState()
start = StartState()
stop = StopState()
context.setState(stop)
context.doThis()
```

## e.g. 计算机系统
以一个计算机系统为例，它可以有多个状态，如开机、关机、挂起或休眠。现在利用状态设计模式来表述这些状态。

首先，我们不妨从`ComputerState`接口开始入手：`state`应定义两个属性，它们是`name`和`allowed`。属性`name`表示对象的状态，而属性`allowed`是定义允许进入的状态的对象的列表；`state`必须定义一个`switch()`方法，由它来实际改变对象的状态：

```python
class ComputerState(object):
    name = "state"
    allowed = []

    def switch(self, state):
        if state.name in self.allowed:
            print('Current:', self, ' => switched to new state', state.name)
            self.__class__ = state
        else:
            print('Current:', self, ' => switching to',
                  state.name, 'not possible.')

    def __str__(self):
        return self.name
```

```python
class Off(ComputerState):
    name = "off"
    allowed = ['on']

class On(ComputerState):
    name = "on"
    allowed = ['off', 'suspend', 'hibernate']

class Suspend(ComputerState):
    name = "suspend"
    allowed = ['on']

class Hibernate(ComputerState):
    name = "hibernate"
    allowed = ['on']
```

```python
class Computer(object):
    def __init__(self, model='HP'):
        self.model = model
        self.state = Off()

    def change(self, state):
        self.state.switch(state)
```

```python
comp = Computer()
# Switch on
comp.change(On)
# Switch off
comp.change(Off)
# Switch on again
comp.change(On)
# Suspend
comp.change(Suspend)
# Try to hibernate - cannot!
comp.change(Hibernate)
# switch on back
comp.change(On)
# Finally off
comp.change(Off)
```

状态设计模式中由于每个状态都是一个类的实例，很容易引起“类爆炸”等问题。
