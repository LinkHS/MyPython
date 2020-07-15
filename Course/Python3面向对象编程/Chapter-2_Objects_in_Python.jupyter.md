# 第2章 Python对象 [Objects in Python]

## e.g. 笔记本
创建一个简单的命令行笔记本应用。笔记是保存在笔记本中的备忘录，每条笔记应该有标签（用于添加和查询）和创建日期，笔记可以被修改，内容也可以被搜索，这些都是通过命令完成。

关于命令行菜单接口的设计，为了让笔记本兼容未来可能添加的新接口（例如GUI工具集或基于Web的接口），所以菜单应该是一个单独的模块。

![](http://static.zybuluo.com/AustinMxnet/rro8z4dfzzbmm48nt71n3tt3/image.png)

下面实现笔记类（`class Note`）：

```python
import datetime

class Note:
    """Represent a note in the notebook. Match against a
    string in searches and store tags for each note.
    """
    def __init__(self, memo, ID, tags=''):
        """Initialize a note with memo and optional
        space-separated tags. Automatically set the note's
        creation date and a unique id.
        """
        self.memo = memo
        self.tags = tags
        self.creation_date = datetime.date.today()
        self.id = ID
    
    def match(self, keyword):
        """Determine if this note matches the keyword
        text. Return True if it matches, False otherwise.
        
        Search is case sensutuve and matches both text and tags.
        """
        return keyword in self.memo or keyword in self.tags
```

测试一下`class Note`的“初始化”和“查找”功能：

```python
n1 = Note('first memo', 0)
n2 = Note('hello world', 1)

print(n1.id, n1.memo, n1.tags)
print(n2.id, n2.memo, n2.tags)
print(n1.match('hello'))
print(n2.match('world'))
```

接着实现笔记本类（`class Notebook`），提供了“新建笔记”、“修改笔记内容/标签”、“查找笔记”的功能：

```python
class Notebook:
    """Represent a collection of notes that can be tagged,
    modified, and searched."""
    
    def __init__(self):
        """Initialize a notebook with an empty list."""
        self.notes = []
    
    def new_note(self, memo, tags=''):
        """Create a new note and add it to the list."""
        self.notes.append(Note(memo, len(self.notes), tags))
    
    def _find_note(self, note_id):
        """Locate the note with the given id"""
        for note in self.notes:
            if note.id == note_id:
                return note
        return None

    def modify_memo(self, note_id, memo):
        """Find the note with the given id and change its
        memo to the given value."""
        self._find_note(note_id).memo = memo
    
    def modify_tags(self, note_id, tags):
        """Find the note with the given id and change its
        tags to the given value."""
        for note in self.notes:
            if note.ID == note_id:
                note.tags = tags
                break
    
    def search(self, keyword):
        """Find all notes that match the given filter string."""
        return [note for note in self.notes if note.match(keyword)]
```

测试`class Notebook`的“新建笔记”功能：

```python
n = Notebook()
n.new_note('hello world')
n.new_note('hello again')

print('notes:', n.notes)
print('id1:', n.notes[0].id, n.notes[0].memo)
print('id2:', n.notes[1].id, n.notes[1].memo)
```

测试`class Notebook`的“修改笔记”功能：

```python
n.modify_memo(1, 'new msg')
n.notes[1].memo
```

尝试修改不存在的笔记：

```python
try:
    n.modify_memo(2, 'sasa')
except Exception as E:
    print("Exception: {}".format(type(E).__name__))
    print("Exception message: {}".format(E))
```

继续完成命令行菜单类`class Menu`：

```python
class Menu:
    """Display a menu and respond to choices when run."""

    def __init__(self):
        self.notebook = Notebook()
        self.choices = {
            "1": self.show_notes,
            "2": self.search_notes,
            "3": self.add_note,
            "4": self.modify_note,
            "5": self.quit
        }

    def display_menu(self):
        print("""
Notebook Menu

1. Show all Notes
2. Search Notes
3. Add Note
4. Modify Note
5. Quit
""")

    def run(self):
        """Display the menu and respond to choices."""
        while True:
            self.display_menu()
            choice = input("Enter an option: ")
            action = self.choices.get(choice)
            if action:
                action()
            else:
                print("{0} is not a valid choice".format(choice))
    
    def show_notes(self, notes=None):
        if not notes:
            notes = self.notebook.notes
        for note in notes:
            print("{0}: {1}\n{2}".format(note.id, note.tags, note.memo))
    
    def search_notes(self):
        keyword = input("Search for: ")
        notes = self.notebook.search(keyword)
        self.show_notes(notes)
    
    def add_note(self):
        memo = input("Enter a memo: ")
        self.notebook.new_note(memo)
        print("Your note has been added.")
    
    def modify_note(self):
        ID = input("Enter a note id: ")
        memo = input("Enter a memo: ")
        tags = input("Enter tags:")
        if memo:
            self.notebook.modify_memo(ID, memo)
        if tags:
            self.notebook.modify_tags(ID, tags)
            
    def quit(self):
        print("Thank you for using your notebook today.")
        raise SystemExit
```

测试一下命令行的菜单功能，需要手动打开`menu.run()`：

```python
menu = Menu()
menu.display_menu()
#menu.run()
```
