# 第1章 Python数据模型
## 一摞Python风格的纸牌

```python
import collections

Card = collections.namedtuple('Card', ['rank', 'suit'])


class FrenchDeck:
    ranks = [str(n) for n in range(2, 11)] + list('JQKA')
    suits = 'spades diamonds clubs hearts'.split()

    def __init__(self):
        self._cards = [Card(rank, suit) for suit in self.suits
                                        for rank in self.ranks]

    def __len__(self):
        return len(self._cards)

    def __getitem__(self, position):
        return self._cards[position]
```

```python
beer_card = Card('7', 'diamonds')
beer_card
```

```python
deck = FrenchDeck()
len(deck)
```

```python
from random import choice
choice(deck)
```

查看最上面3张和只看牌面是A的牌：

```python
deck[:3]
```

```python
deck[12::13]
```

仅仅实现了`__getitem__`方法，这一摞牌就变成可迭代的了：

```python
for i, card in enumerate(reversed(deck)):
    if i <10 or i > 40:
        print(card)
    if i == 11:
        print('......\n......')
```

升序排名：

```python
def spades_high(card):
    """
    1. 点数判定：2 最小、A最大
    2. 花色判定：黑桃最大、红桃次之、方块再次、梅花最小
    """
    rank_value = FrenchDeck.ranks.index(card.rank)
    suit_values = dict(spades=3, hearts=2, diamonds=1, clubs=0)
    return rank_value * len(suit_values) + suit_values[card.suit]
```

```python
i = 0
for card in sorted(deck, key=spades_high):
    if i <10 or i > 47:
        print(card)
    if i == 11:
        print('......\n......')
    i += 1
```

## 如何使用特殊方法
### 模拟数值类型

![image](http://static.zybuluo.com/AustinMxnet/iyahey7fown74bnzvbom0o6r/image.png)

实现一个二维向量（vector）类：

> Python内置的`complex`类可以用来表示二维向量，但我们这个自定义的类可以扩展到$n$维向量，详见第14章。

```python
from math import hypot


class Vector:
    def __init__(self, x=0, y=0):
        self.x = x
        self.y = y

    def __repr__(self):
        return 'Vector(%r, %r)' % (self.x, self.y)

    def __abs__(self):
        return hypot(self.x, self.y)

    def __bool__(self):
        # return bool(self.x or self.y) # 更高效
        return bool(abs(self))

    def __add__(self, other):
        x = self.x + other.x
        y = self.y + other.y
        return Vector(x, y)

    def __mul__(self, scalar):
        return Vector(self.x * scalar, self.y * scalar)
```

加法和模运算：

```python
v1 = Vector(2, 4)
v2 = Vector(2, 1)
v1 + v2
```

```python
v = Vector(3, 4)
abs(v)
```

### 字符串表现形式


在`__repr__`的实现中，**我们用到了`%r`来获取对象各个属性的标准字符串表示形式**---这是个好习惯，它暗示了一个关键：`Vector(1, 2)`和`Vector('1', '2')`是不一样的，后者在我们的定义中会报错，因为向量对象的构造函数只接受数值，不接受字符串。

```python
v1 = Vector(2, 4)
v2 = Vector('2', '4')

print(v1, v2, sep='\n')
```
