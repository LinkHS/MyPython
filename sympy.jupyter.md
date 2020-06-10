# sympy

```python
import numpy as np
from  IPython.display import display, Math, Latex
import sympy as sp
```

```python
raw_latex = "\\text{%s} \quad W*X^T" % ("e.g.")

raw_latex
```

```python
display(Math(raw_latex))
```

```python
display(Latex("$"+raw_latex+"$"))
```

```python
a1 = np.eye(2, 2, dtype=np.int)

a1
```

## 显示`np.array`

```python
raw_latex1 = "\\text{e.g.} \quad %s" % (sp.sympify(a1))
raw_latex2 = "\\text{e.g.} \quad %s" % (sp.latex(sp.sympify(a1)))

raw_latex1, raw_latex2
```

```python
display(Math(sp.latex(raw_latex1)))
display(Math(sp.latex(raw_latex2)))
```

```python

```

```python

```
