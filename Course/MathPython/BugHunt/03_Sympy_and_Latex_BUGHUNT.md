# Sympy and LaTex: Bug hunt!

```python
mu,alpha = sym.symbols('mu,alpha')

expr = 2*sym.exp(mu**2/alpha)

display(Math( expr ))
```

```python
Math('1234 + \frac{3x}{\sin(2\pi t+\theta)}')
```

```python
a = '3'
b = '4'

# answer should be 7
print(sym.sympify(a+b))
```

```python
sym.Eq( 4*x = 2 )
```

```python
# part 1 of 2

q = x^2
r = x**2

display(q)
display(r)
```

```python
# part 2 of 2

q,r = sym.symbols('q,r')

q = sym.sympify('x^2')
r = sym.sympify('x**2')

display(q)
display(r)

sym.Eq(q,r)
```

```python
x = sym.symbols('x')

equation = (4*x**2 - 5*x + 10)**(1/2)
display(equation)
sym.subs(equation,x,3)
```

```python
x,y = sym.symbols('x,y')

equation = 1/4*x*y**2 - x*(5*x + 10*Y**2)**(3)
display(equation)
```
