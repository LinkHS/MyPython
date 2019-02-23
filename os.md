

```
import os

# --- 
# Getting the name of the file without the extension
print(os.path.splitext("/home/austin/temp.md.txt")[0])
>>>
/home/austin/temp.md

# ---
# Get the extension
print(os.path.splitext("/home/austin/temp.md.txt")[1])
>>>
.txt

# ---
print(os.path.basename("/home/austin/temp.md.txt"))
>>>
temp.md.txt

# ---
print(os.path.splitext(os.path.basename("/home/austin/temp.md"))[0])
>>>
temp
```