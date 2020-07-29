# Chapter9 模型-视图-控制器——复合模式
Model-View-Controller (MVC)不仅是一种实现用户界面的软件模式，同时也是一种易于修改和维护的架构。通常来说，MVC模式将应用程序分为3个基本部分：模型、视图和控制器。这3个部分是相互关联的，并且有助于将信息的处理与信息的呈现分离开来。MVC模式的工作机制为：模型提供数据和业务逻辑（如何存储和查询信息），视图负责数据的展示（如何呈现），而控制器是两者之间的粘合剂，根据用户要求的呈现方式来协调模型和视图。有趣的是，**视图和控制器依赖于模型**，而不是反过来，这是因为用户关心的是数据，而模型是可以独立工作的，这是MVC模式的关键所在。

![](http://static.zybuluo.com/AustinMxnet/r2k9064qxsr0adudjtfgofjd/image.png)

```python
class Model(object):
    services = {
        'email': {'number': 1000, 'price': 2, },
        'sms': {'number': 1000, 'price': 10, },
        'voice': {'number': 1000, 'price': 15, },
    }


class View(object):
    def list_services(self, services):
        for svc in services:
            print(svc, ' ')

    def list_pricing(self, services):
        for svc in services:
            print("For", Model.services[svc]['number'],
                  svc, "message you pay $",
                  Model.services[svc]['price'])
```

```python
class Controller(object):
    def __init__(self):
        self.model = Model()
        self.view = View()

    def get_services(self):
        services = self.model.services.keys()
        return(self.view.list_services(services))

    def get_pricing(self):
        services = self.model.services.keys()
        return(self.view.list_pricing(services))
```

```python
controller = Controller()
print("Services Provided:")
controller.get_services()
print("Pricing for Services:")
controller.get_pricing()
```

在UML图中，我们可以看到这个模式中的3个主要类：

- Model：定义与客户端的某些任务有关的业务逻辑或操作。
- View：定义客户端查看的视图或展示。模型根据业务逻辑向视图呈现数据。
- Controller：这实际上是视图和模型之间的接口。当客户端采取某些操作时，控制器将来自视图的查询传递给模型。

![](http://static.zybuluo.com/AustinMxnet/a7qigkwt6jharjdc9cuaq85u/image.png)

## e.g. Web应用程序
Web应用程序框架也是基于MVC的优秀理念的。以Django或Rails（Ruby）为例：它们都是以模式—视图—控制器格式来构造项目的，只是形式为模型、模版、视图（Model-Template-View，MTV），其中模型是数据库，模板是视图，控制器是视图/路由。举例来说，假设要用[Tornado Web应用程序框架](http://www.tornadoweb.org/en/stable/)来开发一个单页应用程序。这个应用程序用于管理用户的各种任务，同时用户还具有添加任务、更新任务和删除任务的权限。

1. 在Tornado中，控制器被定义为视图/应用程序路由。我们需要定义多个视图，例如列出任务、创建新任务、关闭任务，以及在无法处理请求时的操作；
2. 我们还应该定义模型，即列出、创建或删除任务的数据库操作；
3. 最后，视图由Tornado中的模板显示。对于应用程序来说，我们需要一个模板来显示、创建或删除任务，以及另一个模板用于没有找到URL时的情形。

在Tornado中，数据库操作是在不同的处理程序下执行的。处理程序根据用户在Web应用程序中请求的路由对数据库执行操作。在这里讨论的是在这个例子中创建的4个处理程序：
- `IndexHandler`：返回存储在数据库中的所有任务。它返回一个与关键任务有关的字典。它执行SELECT数据库操作来获取这些任务。
- `NewHandler`：顾名思义，它对添加新任务很有用。它检查是否有一个POST调用来创建一个新任务，并在数据库中执行INSERT操作。
- `UpdateHandler`：在将任务标记为完成或重新打开给定任务时非常有用。在这种情况下，将执行UPDATE数据库操作，将任务的状态设置为open / closed。
- `DeleteHandler`：这将从数据库中删除指定的任务。一旦删除，任务将会从任务列表中消失。

```script magic_args="true"

import sqlite3
import tornado.httpserver
import tornado.ioloop
import tornado.web
import tornado


class IndexHandler(tornado.web.RequestHandler):
    def get(self):
        query = "select * from task"
        todos = _execute(query)
        self.render('index.html', todos=todos)


class NewHandler(tornado.web.RequestHandler):
    def post(self):
        name = self.get_argument('name', None)query = "create table if not exists task (id INTEGER \
        PRIMARY KEY, name TEXT, status NUMERIC) "
        _execute(query)
        query = "insert into task (name, status) \
        values ('%s', %d) " % (name, 1)
        _execute(query)
        self.redirect('/')

    def get(self):
        self.render('new.html')


class UpdateHandler(tornado.web.RequestHandler):
    def get(self, id, status):
        query = "update task set status=%d where \
        id=%s" % (int(status), id)
        _execute(query)
        self.redirect('/')


class DeleteHandler(tornado.web.RequestHandler):
    def get(self, id):
        query = "delete from task where id=%s" % id
        _execute(query)
        self.redirect('/')
```

<!-- #region -->
在这个示例中，我们有4个应用程序路由：
- `/`：这用于列出所有任务的路由。
- `/todo/new`：这是创建新任务的路由。
- `/todo/update`：这是将任务状态更新为打开或关闭的路由。
- `/todo/delete`：这是删除已完成任务的路由。

base.html
```html
<html>
<!DOCTYPE>
<html>
<head>{% block header %}{% end %}</head>
<body>{% block body %}{% end %}</body>
</html>
```

index.html
```html
{% extends 'base.html' %}
<title>ToDo</title>
{% block body %}
<h3>Your Tasks</h3>
<table border="1" >
<tralign="center"><td>Id</td>
<td>Name</td>
<td>Status</td>
<td>Update</td>
<td>Delete</td>
<tr>
  {% for todo in todos %}
<tralign="center">
<td>{{todo[0]}}</td>
<td>{{todo[1]}}</td>
    {% if todo[2] %}
<td>Open</td>
    {% else %}
<td>Closed</td>
    {% end %}
    {% if todo[2] %}
<td><a href="/todo/update/{{todo[0]}}/0">Close Task</a></td>
    {% else %}
<td><a href="/todo/update/{{todo[0]}}/1">Open Task</a></td>
  {% end %}
<td><a href="/todo/delete/{{todo[0]}}">X</a></td>
</tr>
{% end %}
</table>
<div>
<h3><a href="/todo/new">Add Task</a></h3>
</div>
{% end %}
```

new.html
```html
{% extends 'base.html' %}
<title>ToDo</title>
{% block body %}
<div>
<h3>Add Task to your List</h3>
<form action="/todo/new" method="post" id="new">
<p><input type="text" name="name" placeholder="Enter task"/>
<input type="submit" class="submit" value="add" /></p>
</form>
</div>
{% end %}
```
<!-- #endregion -->

```script magic_args="true"

class RunApp(tornado.web.Application):
    def __init__(self):
        Handlers = [
            (r'/', IndexHandler),
            (r'/todo/new', NewHandler),
            (r'/todo/update/(\d+)/status/(\d+)', UpdateHandler),
            (r'/todo/delete/(\d+)', DeleteHandler)]

        settings = dict(
            debug=True,
            template_path='templates',
            static_path="static",
        )
        tornado.web.Application.__init__(self, Handlers, **settings)
```

```script magic_args="true"

http_server = tornado.httpserver.HTTPServer(RunApp())
http_server.listen(5000)
tornado.ioloop.IOLoop.instance().start()
```

当我们运行这个Python程序时：
1. 服务器将启动，并在端口5000上运行，适当的视图、模板和控制器已经配置好了；
2. 浏览http:// localhost:5000 /，可以看到任务列表。

由于需要安装tornado和sqlite3，这里就没有实际运行，运行结果如下图所示：

![](http://static.zybuluo.com/AustinMxnet/4wfg8e59fdmvg06dtamry5rz/image.png)
