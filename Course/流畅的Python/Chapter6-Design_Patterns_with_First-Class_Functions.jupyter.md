<!-- #region -->
# 使用一等函数实现设计模式
## 案例分析：重构“策略”模式
### 经典的“策略”模式
UML类图指出了“策略”模式对类的编排：

![image](http://static.zybuluo.com/AustinMxnet/c5twtolnpa78ha5nnc3dvygb/image.png)

假如一个网店制定了下述折扣规则：
- 有 1000 或以上积分的顾客，每个订单享 5% 折扣。
- 同一订单中，单个商品的数量达到 20 个或以上，享 10% 折扣。
- 订单中的不同商品达到 10 个或以上，享 7% 折扣。


#### e.g. 实现`Order`类，支持插入式折扣策略
<!-- #endregion -->

```python
from abc import ABC, abstractmethod
from collections import namedtuple

Customer = namedtuple('Customer', 'name fidelity')


class LineItem:
    def __init__(self, product, quantity, price):
        self.product = product
        self.quantity = quantity
        self.price = price

    def total(self):
        return self.price * self.quantity


class Order:  # the Context
    def __init__(self, customer, cart, promotion=None):
        self.customer = customer
        self.cart = list(cart)
        self.promotion = promotion

    def total(self):
        if not hasattr(self, '__total'):
            self.__total = sum(item.total() for item in self.cart)
        return self.__total

    def due(self):
        if self.promotion is None:
            discount = 0
        else:
            discount = self.promotion.discount(self)
        return self.total() - discount

    def __repr__(self):
        fmt = '<Order total: {:.2f} due: {:.2f}>'
        return fmt.format(self.total(), self.due())
```

下面先定义了一个抽象类`Promotion`，然后分别实现了三个策略类：

```python
from abc import ABC, abstractmethod
from collections import namedtuple

Customer = namedtuple('Customer', 'name fidelity')


class LineItem:
    def __init__(self, product, quantity, price):
        self.product = product
        self.quantity = quantity
        self.price = price

    def total(self):
        return self.price * self.quantity


class Order:  # the Context
    def __init__(self, customer, cart, promotion=None):
        self.customer = customer
        self.cart = list(cart)
        self.promotion = promotion

    def total(self):
        if not hasattr(self, '__total'):
            self.__total = sum(item.total() for item in self.cart)
        return self.__total

    def due(self):
        if self.promotion is None:
            discount = 0
        else:
            discount = self.promotion.discount(self)
        return self.total() - discount

    def __repr__(self):
        fmt = '<Order total: {:.2f} due: {:.2f}>'
        return fmt.format(self.total(), self.due())


class Promotion(ABC):  # the Strategy: an abstract base class
    @abstractmethod
    def discount(self, order):
        """Return discount as a positive dollar amount"""


class FidelityPromo(Promotion):  # first Concrete Strategy
    """5% discount for customers with 1000 or more fidelity points"""

    def discount(self, order):
        return order.total() * .05 if order.customer.fidelity >= 1000 else 0


class BulkItemPromo(Promotion):  # second Concrete Strategy
    """10% discount for each LineItem with 20 or more units"""

    def discount(self, order):
        discount = 0
        for item in order.cart:
            if item.quantity >= 20:
                discount += item.total() * .1
            return discount


class LargeOrderPromo(Promotion):  # third Concrete Strategy
    """7% discount for orders with 10 or more distinct items"""

    def discount(self, order):
        distinct_items = {item.product for item in order.cart}
        if len(distinct_items) >= 10:
            return order.total() * .07
        return 0
```

两个顾客：joe的积分是0，ann的积分是1100，  
有三个商品的购物车：

```python
joe = Customer('John Doe', 0)
ann = Customer('Ann Smith', 1100)
cart = [LineItem('banana', 4, .5),
        LineItem('apple', 10, 1.5),
        LineItem('watermellon', 5, 5.0)]
```

`FidelityPromo`没joe提供折扣：

```python
Order(joe, cart, FidelityPromo())
```

ann得到了5%折扣，因为她的积分超过1000：

```python
Order(ann, cart, FidelityPromo())
```

`banana_cart`中有30把香蕉和10个苹果，  
`BulkItemPromo`为joe购买的香蕉优惠了1.50美元：

```python
banana_cart = [LineItem('banana', 30, .5),
               LineItem('apple', 10, 1.5)]

Order(joe, banana_cart, BulkItemPromo())
```

`long_order`中有10个不同的商品，每个商品的价格为1.00美元：

```python
long_order = [LineItem(str(item_code), 1, 1.0) for item_code in range(10)]

Order(joe, long_order, LargeOrderPromo())
```

LargerOrderPromo为joe的整个订单提供了7%折扣：

```python
Order(joe, cart, LargeOrderPromo())
```

上面的示例完全可用，但是利用Python中作为对象的函数，可以使用更少的代码实现相同的功能，详情参见下一节。

### 使用函数实现“策略”模式
上一节示例中，每个具体策略都是一个类，而且都只定义了一个方法，即`discount`。此外， 策略实例没有状态（没有实例属性）。你可能会说，它们看起来像是普通的函数。

#### e.g. `Order`类和使用函数实现的折扣策略

```python
from collections import namedtuple

Customer = namedtuple('Customer', 'name fidelity')


class LineItem:
    def __init__(self, product, quantity, price):
        self.product = product
        self.quantity = quantity
        self.price = price

    def total(self):
        return self.price * self.quantity


class Order:  # the Context
    def __init__(self, customer, cart, promotion=None):
        self.customer = customer
        self.cart = list(cart)
        self.promotion = promotion

    def total(self):
        if not hasattr(self, '__total'):
            self.__total = sum(item.total() for item in self.cart)
        return self.__total

    def due(self):
        if self.promotion is None:
            discount = 0
        else:
            discount = self.promotion(self)
        return self.total() - discount

    def __repr__(self):
        fmt = '<Order total: {:.2f} due: {:.2f}>'
        return fmt.format(self.total(), self.due())
```

下面把具体策略换成了简单的函数，而且去掉了`Promo`抽象类：

```python
def fidelity_promo(order):  # first Concrete Strategy
    """5% discount for customers with 1000 or more fidelity points"""
    return order.total() * .05 if order.customer.fidelity >= 1000 else 0

def bulk_item_promo(order):  # second Concrete Strategy
    """10% discount for each LineItem with 20 or more units"""
    discount = 0
    for item in order.cart:
        if item.quantity >= 20:
            discount += item.total() * .1
        return discount


def large_order_promo(order):  # third Concrete Strategy
    """7% discount for orders with 10 or more distinct items"""
    distinct_items = {item.product for item in order.cart}
    if len(distinct_items) >= 10:
        return order.total() * .07
    return 0
```

代码少了12 行，新的`Order`类使用起来更简单：
- 计算折扣只需调用`self.promotion()`函数
- 没有抽象类
- 各个策略都是函数

测试代码几乎完全一样，但是少了在新建订单时实例化新的促销对象，只把促销函数作为参数传入：

```python
joe = Customer('John Doe', 0)
ann = Customer('Ann Smith', 1100)
cart = [LineItem('banana', 4, .5),
        LineItem('apple', 10, 1.5),
        LineItem('watermellon', 5, 5.0)]

Order(joe, cart, fidelity_promo)
```

## “命令”模式
“命令”设计模式也可以通过把函数作为参数传递而简化。这一模式对类的编排如图

```python

```

```python

```

```python

```
