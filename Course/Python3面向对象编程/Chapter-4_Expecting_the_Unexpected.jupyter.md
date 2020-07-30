# 第4章 异常捕获
## e.g.  中央认证和授权系统
“认证”是确保当前用户是其本人，使用“用户名”和“密码”组合进行验证。“授权”则是决定一个用户是否有权执行某项操作，可以用一个许可列表系统，存储各项操作允许的特定用户列表。

首先我们需要一个`User`类保存用户名和加密的密码，同时提供接口检查密码是否正确：

```python
import hashlib


class User:
    def __init__(self, username, password):
        '''Create a new user object. The password
        will be encrypted before storing.'''
        self.username = username
        self.password = self._encrypt_pw(password)
        self.is_logged_in = False

    def _encrypt_pw(self, password):
        '''Encrypt the password with the username and return
        the sha digest.'''
        hash_string = (self.username + password)
        hash_string = hash_string.encode("utf8")
        return hashlib.sha256(hash_string).hexdigest()

    def check_password(self, password):
        '''Return True if the password is valid for this
        user, false otherwise.'''
        encrypted = self._encrypt_pw(password)
        return encrypted == self.password
```

```python
Rachel = User("Rachel", "Password123")

print(Rachel.check_password("Pass"))
print(Rachel.check_password("Password123"))
```

然后需要一个`Authenticator`类，用于管理用户，例如添加用户、用户登入和登出。但是管理过程中会遇到很多异常，例如“密码错误”、“用户名不存在”等。我们定义一些异常类，方便未来扩展：

```python
class AuthException(Exception):
    def __init__(self, username, user=None):
        super().__init__(username)
        self.username = username
        self.user = user


class UsernameAlreadyExists(AuthException):
    pass


class PasswordTooShort(AuthException):
    pass


class InvalidUsername(AuthException):
    pass


class InvalidPassword(AuthException):
    pass
```

异常类使用示例：

```python
def expect_exception(cmd):
    """防止Jupyter Notebook自动执行时遇到Exception停止
    """
    try:
        eval(cmd)
    except Exception as E:
        print("Exception: {}".format(type(E).__name__))
        print("Exception message: {}".format(E))


def show_exception():
    # `eval()`不支持传入`raise`开头的命令，用本函数包装一下
    raise UsernameAlreadyExists("Rachel")

expect_exception('show_exception()')
del show_exception # 删除临时测试函数
```

添加一个`Authenticator`类，用于管理用户，例如添加用户、用户登入和登出：

```python
class Authenticator:
    def __init__(self):
        '''Construct an authenticator to manage
        users logging in and out.'''
        self.users = {}

    def add_user(self, username, password):
        if username in self.users:
            raise UsernameAlreadyExists(username)
        if len(password) < 6:
            raise PasswordTooShort(username)
        self.users[username] = User(username, password)

    def login(self, username, password):
        try:
            user = self.users[username]
        except KeyError:
            raise InvalidUsername(username)

        if not user.check_password(password):
            raise InvalidPassword(username, user)

        user.is_logged_in = True
        return True

    def is_logged_in(self, username):
        if username in self.users:
            return self.users[username].is_logged_in
        return False
```

```python
authenticator = Authenticator()
authenticator.add_user("Rachel", "Password123")
print("---再次尝试加入Rachel---")
expect_exception('authenticator.add_user("Rachel", "")')

authenticator.login("Rachel", "Password123")
print("\n---尝试错误密码登录---")
expect_exception('authenticator.login("Rachel", "Pass")')
```

最后添加`Authorizor`，用于授权和检查用户是否可以执行某项操作：

```python
class PermissionError(Exception):
    pass


class NotLoggedInError(AuthException):
    pass


class NotPermittedError(AuthException):
    pass


class Authorizor:
    def __init__(self, authenticator):
        self.authenticator = authenticator
        self.permissions = {}

    def add_permission(self, perm_name):
        '''Create a new permission that users can be added to'''
        try:
            perm_set = self.permissions[perm_name]
        except KeyError:
            self.permissions[perm_name] = set()
        else:
            raise PermissionError("Permission Exists")

    def permit_user(self, perm_name, username):
        '''Grant the given permission to the user'''
        try:
            perm_set = self.permissions[perm_name]
        except KeyError:
            raise PermissionError("Permission does not exist")
        else:
            if username not in self.authenticator.users:
                raise InvalidUsername(username)
            perm_set.add(username)

    def check_permission(self, perm_name, username):
        try:
            perm_set = self.permissions[perm_name]
        except KeyError:
            raise PermissionError("Permission does not exist")
        else:
            if username not in perm_set:
                raise NotPermittedError(username)
            else:
                return True
```

```python
authenticator = Authenticator()
authorizor = Authorizor(authenticator)

authenticator.add_user("Rachel", "Password123")

authorizor.add_permission("Restart")
authorizor.permit_user("Restart", "Rachel")
authorizor.check_permission("Restart", "Rachel")
```
