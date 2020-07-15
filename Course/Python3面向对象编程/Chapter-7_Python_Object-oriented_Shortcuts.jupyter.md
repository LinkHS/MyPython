# 第7章 Python面向对象的捷径
## Python内置函数
### `reversed`

```python
normal_list=[1,2,3,4,5]
[item for item in reversed(normal_list)]
```

### `enumerate`

```python
from operator import itemgetter

def min_max_indexes(seq):
    minimum = min(enumerate(seq), key=itemgetter(1))
    maximum = max(enumerate(seq), key=itemgetter(1))
    return minimum[0], maximum[0]

print([i for i in enumerate(normal_list)])
min_max_indexes([1, 2, 3, 4])
```

#### `__enter__`

```python
import string
import random


class StringJoiner(list):
    def __enter__(self):
        print('enter', self)
        self.append(0)
        return self

    def __exit__(self, type, value, tb):
        print('exit', self)


with StringJoiner() as joiner:
    joiner.append(1)

joiner
```

## e.g. 邮件列表管理器
为了将本章所讲的知识点整理到一起，我们来建造一个邮件列表管理器。这个管理器将会追踪归类到不同组中的邮箱地址。当需要发送信息时，可以挑选一个组并向该组中的所有地址发送信息。

在`send_email()`函数调用中，同时用到了变量参数和关键字参数语法。变量参数列表让我们既可以按照默认情况提供一个单独的字符串作为`to_addrs`，也可以允许传递多个地址。

```python
from collections import defaultdict

def send_email(subject, message, from_addr, *to_addrs):
    print(subject, message)
    print('from:', from_addr)
    print('to:', *to_addrs)
```

```python
from_addr = "123@qq.com"
to_addrs = ["111@qq.com", "112@qq.com"]
send_email('Invitation:', 'Have dinner together.', from_addr, *to_addrs)
```

它已经将我们的邮件主题和内容“发送”给两个期望的地址。现在开始准备邮件组的管理系统：需要一个对象来将邮箱地址匹配到所属的组。由于这是一个多对多的关系（任意一封邮件都可以属于多个组，任何一个组都可以包含多个邮箱地址），现有的Python数据结构都不够理想：1. 用字典存储，将组名匹配到相关邮箱地址的列表，但是这样就会存在很多重复的邮箱地址；2. 将邮箱地址字典匹配到组，这样又会出现重复的组。它们看起来都不是最优方案，暂且先尝试后一个方案（尽管组名指向邮箱地址可能更直观）。

由于字典的值将会是一些唯一邮箱地址的集合，我们将它们存储到`set`容器中，**可以用`defaultdict`来确保每个键都有对应的`set`容器**：

> `__enter__`和`__exit__`是为了安全地操作文件，由于展示原因，省略了`save`和`load`中对文件操作的代码。

```python
addresses_db = ["friend1@example.com friends",
                "family1@example.com family,friends"]

class MailingList:
    '''Manage groups of e-mail addresses for sending e-mails.'''

    def __init__(self, data_file):
        self.data_file = data_file
        self.email_map = defaultdict(set)

    def add_to_group(self, email, group):
        self.email_map[email].add(group)

    def emails_in_groups(self, *groups):
        groups = set(groups)
        emails = set()
        for e, g in self.email_map.items():
            if g & groups:
                emails.add(e)
        return emails

    def send_mailing(self, subject, message, from_addr, *groups):
        emails = self.emails_in_groups(*groups)
        send_email(subject, message, from_addr, *emails)

    def save(self):
        self.data_file.clear()
        for email, groups in self.email_map.items():
            self.data_file.append('{} {}\n'.format(email, ','.join(groups)))

    def load(self):
        self.email_map = defaultdict(set)
        for line in self.data_file:
            email, groups = line.strip().split(' ')
            groups = set(groups.split(','))
            self.email_map[email] = groups

    def __enter__(self):
        self.load()
        return self

    def __exit__(self, type, value, tb):
        self.save()
```

读取`addresses_db`，添加`friend2`地址，发送邮件给多个邮箱：

```python
from pprint import pprint

with MailingList(addresses_db) as ml:
    print('email_map:')
    pprint(ml.email_map)
    
    ml.add_to_group('friend2@example.com', 'friends')
    print('\n---sending---')
    ml.send_mailing("What's up", "hey friends, how's it going", 'me@example.com', 'friends')

print('\naddresses_db:')
pprint(addresses_db)
```
