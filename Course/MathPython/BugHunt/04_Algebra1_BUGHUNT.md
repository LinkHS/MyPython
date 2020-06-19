# BUG HUNT!


```python
from sympy.abc import x2

x2 = 4

```


```python
a,b,c = sym.symbols('a,b,c')

expr = 4*b + 5*a*a - c**3 + 5*d

```


```python
import math
gcd(30,50)
```


```python
expr = 4*x - 8
solve(expr)
```


```python
import numpy as np

A = np.array( [ [1,2],[3,4] ] )
# make it look nice
A
```


```python
fact_dict = sym.factorint(44)
allkeys = fact_dict.keys()

for i in range(0,len(allkeys)):
    print('%g was present %g times.' %(i,allkeys[i]))

```


```python
x,y = sym.symbols('x,y')

expr = 4*x - 5*y**2

expr.subs({x=5})

```


```python
# goal is to show a fraction

f = 5/9

display(Math(sym.latex(f)))

```


```python
# print the last 3 items from a list
lst = [1,3,2,5,4,6,7,5,3,7]
lst[-3:-1]

```


```python
from sympy.abc import x,y

expr = 2*x + 4*y

# solve for y
sym.solve(expr)
```


```python
import numpy as np

A = np.array( [ [1,2],[3,4] ] )

# set the element in the second row, second column to 9
A[2,2] = 9
print(A)
```
