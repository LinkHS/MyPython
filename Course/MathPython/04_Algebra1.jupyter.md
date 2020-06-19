# Algebra 1

```python
# It's generally good practice to import all required modules at the top of the script!
import sympy as sym
import numpy as np
from IPython.display import display, Math
```

## Solving for x

```python
x = sym.symbols('x')

# the expression we want to solve is 2x+4=9
expr = 2*x + 4 -9

sym.solve(expr,x)
```

```python
# make it look a bit nicer

sol = sym.solve(expr,x)

display('The solution to %s is %g'%(expr,sol[0]))

# or
display(Math('\\text{The solution to }%s\\text{ is x=}%g' %(sym.latex(expr),sol[0])))
```

```python
# can input the equation directly into the solve function
sym.solve(x**2 - 4,x)
```

```python
# notice the solution is stored as a list, with one solution per element
sol = sym.solve(x**2 - 4,x)

print( type(sol) )
print( len(sol) )
```

```python
# we can print them all out:
for i in range(0,len(sol)):
    print('Solution #' + str(i+1) + ' is ' + str(sol[i]))

```

```python
y = sym.symbols('y')

expr = x/4 - x*y + 5

print( "Solved for x: " + str(sym.solve(expr,x)[0]) )
print( "Solved for y: " + str(sym.solve(expr,y)) )

```

### Exercises

```python
# 1)
# simplify and solve for q
q = sym.symbols('q')
eq = 3*q + 4/q + 3 - 5*q - 1/q - 1

display(Math(sym.latex(eq.simplify()))) 
display(Math('q='+sym.latex(sym.solve(eq,q))))
```

```python
# 2)
eq = 2*q + 3*q**2 - 5/q - 4/q**3

display(Math(sym.latex(eq)))
display(Math(sym.latex(sym.simplify(eq))))
display(Math(sym.latex(sym.cancel(eq)))) # puts into p/q form with integer coefficients
```

```python
# 3)
# simplify this expression. confirm on your own using paper-and-pencil
expr = (sym.sqrt(3) + sym.sqrt(15)*q) / (sym.sqrt(2) + sym.sqrt(10)*q)
display(Math(sym.latex(expr)))
display(Math(sym.latex(sym.simplify(expr))))
```

```python
sym.simplify( expr.subs(q,10) )
```

```python
expr.subs(q,10).evalf()
```

## Expanding terms

```python
# define our terms
from sympy.abc import x

term1 = (4*x + 5)
term2 = x

print( term1*term2 )
print( sym.expand(term1*term2) )
print( Math(sym.latex(sym.expand(term1*term2)) ))
```

```python
term3 = x - 7 # note that parentheses are not necessary!

display(Math( sym.latex(term1*term3) ))
display(Math( sym.latex( sym.expand(term1*term3) )))
```

```python
# with two variables
y = sym.symbols('y')

expr = x*(2*y**2 - 5**x/x)
sym.expand(expr)
```

```python
# three expressions and three variables!!
# but first, what variables have we already created??
%whos

```

```python
z = sym.symbols('z')

term1 = (3 + x)
term2 = (y - 4*z)
term3 = (5/z + 3*x)

display(Math(sym.latex(term1*term2*term3)))
display(Math(sym.latex(sym.expand(term1*term2*term3))))
display(Math(sym.latex(sym.simplify(sym.expand(term1*term2*term3)))))
```

### Exercises

```python
# a function of two variables
Fxy = (4+x)*(2-y)
print(Fxy.subs({x:2,y:-2}))
```

```python
numrange = range(0,3)
for i in numrange:
    for j in numrange:
        print('When x=%g and y=%g, f(x,y)=%g' %(i,j,Fxy.subs({x:i,y:j})) )
```

## Creating and accessing matrices with numpy

```python
A = np.array( [ [1,2],[3,4] ] )
print(A)

# make it look nicer
display(Math(sym.latex(sym.sympify(A))))
```

```python
# initializing a matrix with zeros

numrange = range(0,5)

mat = np.zeros([len(numrange),len(numrange)])
print(mat)
```

```python
# populating matrices using row-col indexing

mat[0,1] = 1
# mat[5,8] = 4
mat
```

```python
# can also use variables for indices
i = 2
j = 1
mat[i,j] = 4.5

display(Math(sym.latex(sym.sympify(mat))))
```

```python
# now use a for-loop

numrange = range(0,3)
for i in numrange:
    for j in numrange:
        mat[i][j] = (-1)**(i+j)

mat
```

### Exercise

```python
x,y = sym.symbols('x y')
Fxy = (4+x)*(2-y)

numrange = range(0,3)

funout = np.zeros((len(numrange),len(numrange)))

for i in numrange:
    for j in numrange:
        funout[i,j] = Fxy.subs({x:i,y:j})

display(Math(sym.latex(sym.sympify(funout))))
```

### Exercise: Create a multiplication table

```python
nums = range(1,11)

multmat = np.zeros((len(nums),len(nums)),dtype=int)

for i in nums:
    for j in nums:
        multmat[i-1,j-1] = i*j

        
display(Math(sym.latex(sym.sympify(multmat)))) # no display without display

x = 3
```

## Associative, commutative, and distributive properties


### Associative

```python
from sympy.abc import x,y

expr1 = x*(4*y)
expr2 = (x*4)*y

# show that two equations are equal by subtracting them!
expr1 - expr2
```

### Commutative

```python
# create three expressions
e1 = x*4*y
e2 = 4*x*y
e3 = y*x*4
```

```python
# quick reminder about substitution in sympy
display( e1.subs(x,3) )

# multiple subsitutions
e3.subs({x:2,y:3})
```

```python
# now back to the task!
print( e1.subs({x:3,y:4}) )
print( e2.subs({x:3,y:4}) )
print( e3.subs({x:3,y:4}) )
```

### Distributive

```python
# another way of creating symbolic variables
from sympy.abc import a, b, c, d

expr = (a+b)*(c+d)
expr
```

```python
sym.expand(expr)
```

```python
sym.expand( (a+d)*(a-d) )
```

```python
# embedding expressions
a,x,y,z = sym.symbols('a,x,y,z')

x = 3*y + z
a = 4*x

display(a)
```

### Exercises

```python
# with these two expressions, show that the commutative rule applies

w,x,y,z = sym.symbols('w,x,y,z')

x = w*(4-w)+1/w**2*(1+w)
expr1 = x*(y+z)
expr2 = 3/x+x**2

display(Math(sym.latex(expr1*expr2)))
display(Math(sym.latex(sym.simplify(expr1*expr2))))
display(Math(sym.latex(expr2*expr1 - expr1*expr2)))

```

## Creating and working with Python lists

```python
# a list is a collection of things, and created using brackets []
# A simple example is a list of numbers

lst = [1,3,4,7]
print( lst )
print(type(lst))
```

```python
# you can access individual list elements
lst[2]
```

```python
# -1 is for the final element
lst[-1]
```

```python
# "slicing"

# print the first N list items
N = 2
lst[:N]

```

```python
# print the last k items
k = 2
lst[-k:]

```

```python
# print items n through k
lst = [1,2,3,4,5,6,7,8,9]

n = 3
k = 7

lst[n-1:k]

```

```python
# a list of strings
name = ['hello','my','name','is','Mike']

# access each element using a for-loop
for i in range(len(name)):
    print(name[i])
```

```python
# simpler!
for i in name:
    print(i)
```

```python
# lists can also hold more than one variable type
alist = [1,2,'cookies',[4,5]]

for i in alist:
    print(i)
```

```python
# getting items from a list-within-a-list

print( alist[-1] )

print( alist[-1][1] )
```

```python
# the term 'list' is reserved:
alist2 = list( (1,2,'cookies',[4,5]) )

for i in alist2:
    print(i)
```

```python
# importantly, we will use lists for sympy expressions!
# list_of_expressions
expr_list = [ 4*x**2 , 3+x , (1-x)/(1+x) ]
```

### Exercises

```python
# use sympy to expand and simplify these expressions
x = sym.symbols('x')

e1 = 2*x + x*(4-6*x) + x
e2 = -x * (2/x + 4/(x**2)) + (4+x)/(4*x)
e3 = (x + 3)*(x-3)*x*(1/(9*x))

# make a list of the expressions
exprs = [e1,e2,e3]

for i in range(0,3):
    display(Math('%s \\quad \\Longleftrightarrow \\quad %s' %(sym.latex(exprs[i]),sym.latex(sym.expand(exprs[i])))))

```

## More on "slicing" in Python

```python
# create an array (vector) of numbers
vec = [10,11,12,13,14,15,16,17,18,19,20]
# or
vec = list(range(10,21))

print(vec)
```

```python
# indexing a single item
vec[2]
```

```python
# indexing multiple items (aka slicing)
vec[2:4]
```

```python
# from one element to the end
vec[4:]
```

```python
# from the first element to a specific element
vec[:3]
```

```python
# the last element
vec[-1]

# penultimate element
vec[-2]

```

```python
# from the end to the beginning
vec[::-1]
```

```python
# with non-1 stepping
vec[0:5:2]
```

## Greatest common denominator

```python
# reminder: GCD is the largest integer that can divide both numbers without a remainder

# we'll use the math package for this!
import math

math.gcd(95,100)
```

```python
math.gcd(0,3)
```

```python
# application: reduce fraction to lowest term

a = 16
b = 88

fact = math.gcd(a,b)

display(Math('\\frac{%g}{%g} \\quad = \\quad \\frac{%g}{%g} \\times \\frac{%g}{%g}' %(a,b,a/fact,b/fact,fact,fact)))

```

### Exercises

```python
# show this property using symbols, and give an example with numbers.

# gcd(m·a, m·b) = m·gcd(a, b)

a,b,c = sym.symbols('a,b,c')

display( sym.gcd(c*a,c*b) )
display( c*sym.gcd(a,b) )


# now with real numbers
a = 5
b = 6
c = 7
display( math.gcd(c*a,c*b) )
display( c*math.gcd(a,b))

```

```python
# double loop and store in matrix
import numpy as np

N = 10
gcdMat = np.zeros((10,15))+99

for i in range(0,10):
    for j in range(0,15):
        gcdMat[i,j] = math.gcd(i+1,j+1)
        
display(Math(sym.latex(sym.sympify(gcdMat))))
```

## Introduction to dictionaries

```python
# create a dictionary
D = dict(fruit=['banana','apple'],numbers=[1,3,4,2,5])

print(D)
```

```python
# list the "keys"
D.keys()
```

```python
# get the information from the numbers
D['numbers']
```

```python
# or this way
D.get('fruit')[0]
```

```python
len(D)
```

```python
# print out all information in a loop!
for items in D.keys(): # .keys() is implied!
    print(D[items])
```

```python
# make a dictionary of equations
x,y = sym.symbols('x,y')

D = dict(eqsWithX=[x/3-6,x*2+3],eqsWithY=[y-y/4,y-5])
D.keys()

Dkeys = list(D)

# access individual keys
D[Dkeys[0]]

```

### Exercise

```python
# let's make a new dictionary
x,y = sym.symbols('x,y')

# count number of x's and y's in the equation
D = dict(eqsWithX=[4*x-6,x**2-9],eqsWithY=[sym.sin(y)])

# solve them in a loop
for keyi in D:
    
    print('Equations solving for ' + keyi[-1] + ':')
    
    for i in D[keyi]:
        
        fullEQ     = sym.latex(sym.sympify(i)) + ' = 0'
        middlepart = '\\quad\\quad \\Rightarrow\\quad\\quad ' + keyi[-1] + ' = '
        soln       = sym.latex(sym.solve(i))
        
        display(Math( '\\quad\\quad ' + fullEQ + middlepart + soln ))


```

## Prime factorization

```python
# factor an integer into the product of prime numbers
number = 48

# Use the sympy function factorint. The output is a dictionary!
fact_dict = sym.factorint(number)
print(fact_dict)
```

```python
# just print the prime numbers
primenumbers = list( fact_dict.keys() )

print('The prime factors of ' + str(number) + ' are ' + str(primenumbers))

fact_dict[primenumbers[1]]
```

```python
# test on prime number
sym.factorint(4)
```

### Exercise

```python
# loop through numbers and report whether each number is composite or prime

nums = range(2,51)
for i in nums:
    di = sym.factorint(i)
    ks = list(di.keys())
    if len(di)==1 and di[ks[0]]==1:
        print('%s is a prime number' %i)
    else:
        print('%s is a composite number with prime factors %s' %(i,list(di.keys())))

```

## Solving inequalities

```python
x = sym.symbols('x')

expr = 4*x > 8
sym.solve(expr)
```

```python
display(Math(sym.latex(sym.solve(expr))))
```

```python
sym.oo > 10000093847529345
```

```python
expr = (x-1)*(x+3) > 0

display(Math(sym.latex(sym.solve(expr))))
```

```python
# sym.solve will return the expression if not enough information

a,b,c = sym.symbols('a,b,c')

expr = a*x > b**2/c
display(Math(sym.latex(expr)))

try:
    sym.solve(expr)#,x
except Exception as E:
    print("Exception: {}".format(type(E).__name__), E)
```

```python
# a slightly richer problem
sym.solve( 2*x**2>8 )
```

### Exercise

```python
expr = (3*x/2) + (4-5*x)/3 <= 2 - (5*(2-x))/4

display(Math(sym.latex(expr)))
sym.solve(expr)
```

## Adding polynomials

```python
from sympy.abc import x

# straight-forward version
p1 = 2*x**3 + x**2 - x
p2 = x**3 - x**4 - 4*x**2
print( p1+p2 )

display(Math('(%s) + (%s) \quad=\quad (%s)' %(sym.latex(p1),sym.latex(p2),sym.latex(p1+p2) )))
```

```python
# Using the Poly class
p1 = sym.Poly(2*x**6 + x**2 - x)

p1
```

```python
# can implement several methods on the polynomial object
print( p1.eval(10) )

print( p1.degree() )
```

```python
# create a second polynomial
p2 = sym.Poly(x**3 - x**4 - .4*x**2)
print( p1-p2 )

# can also call the add method on the polynomial objects
p1.add(p2)
p1.sub(p2)
print(p1.sub(p2))
print(p1)

```

### Exercise

```python
# create a list of polynomials
# loop through. if order is even, sum the coeffs. if order is odd, count the number of coeffs

polys = [ sym.Poly(2*x + x**2), sym.Poly(-x**3 + 4*x), sym.Poly(x**5-x**4+1/4*x+4) ]

for poli in polys:
    if poli.degree()%2==0:
        print('The degree of %s is even, and the coefficients sum to %s.' %(poli.as_expr(),sum(poli.coeffs())))
    else:
        print('The degree of %s is odd, and there are %s coefficients.' %(poli.as_expr(),len(poli.coeffs())))
```

## Multiplying polynomials

```python
x = sym.symbols('x')

x**2 * x**3
```

```python
# a litte more complicated
p1 = 4*x**2 - 2*x
p2 = x**3 + 1

p1*p2
```

```python
# the way your math teacher would want it written out
print( sym.expand( p1*p2 ) )

display(Math(sym.latex(p1*p2)))
display(Math(sym.latex(sym.expand(p1*p2))))

```

```python
# check our work from the slides!
x,y = sym.symbols('x,y')

poly1 = x**5 + 2*x*y + y**2
poly2 = x - 3*x*y

poly1*poly2
```

```python
display(Math(sym.latex(sym.expand( poly1*poly2 ))))
```

### Exercise

```python
# with x's and y's, substitute before vs after multiplication
x,y = sym.symbols('x,y')

fxy = 4*x**4 - 9*y**3 - 3*x**2 + x*y**2
gxy = 4/5*y**3 - x**3 + 6*x**2*y

display(Math( '(%s)\quad\\times\quad(%s) \quad=\quad %s' %(sym.latex(fxy),sym.latex(gxy),sym.latex(sym.expand(fxy*gxy)) )))

```

```python
xval = 5
yval = -2

# first substitute and then multiply
fxy_subs = fxy.subs({x:xval,y:yval})
gxy_subs = gxy.subs({x:xval,y:yval})
print('Separate substitution: %s' %(fxy_subs*gxy_subs))

# multiply then substitute
fg = (fxy*gxy).subs({x:xval,y:yval})
print('Multiplied substitution: %s' %fg)

```

## Dividing by polynomials

```python
p1 = 4*x**5 - x
p2 = 2*x**3-x

display(Math('\\frac{%s}{%s} = %s' %(sym.latex(p1),sym.latex(p2),sym.latex(p1/p2)) ))
display(Math('\\frac{%s}{%s} = %s' %(sym.latex(p1),sym.latex(p2),sym.latex(sym.expand(p1/p2))) ))
display(Math('\\frac{%s}{%s} = %s' %(sym.latex(p1),sym.latex(p2),sym.latex(sym.simplify(p1/p2))) ))

```

```python
# with two variables
x,y = sym.symbols('x,y')

pNum = x**3 + y**2 - 4*x**2*y + x*y + 4*y
pDen = x + y

display(Math('\\frac{%s}{%s} = %s' %(sym.latex(pNum),sym.latex(pDen),sym.latex(sym.simplify(pNum/pDen))) ))

```

### Exercise

```python
# first, a primer on sym.fraction

num = sym.sympify(3)/sym.sympify(4)
# num = sym.sympify(3/4)

finfo = sym.fraction(num)
print(type(finfo))
print(finfo[0])

# can also isolate the numerator separately
num = sym.fraction(num)[0]
print(num)
```

```python

# use a loop to find the integer value of y that makes this equation simplify
pNum = x**6 + 2*x**4 + 6*x  - y
pDen = x**3 + 3


for i in range(5,16):
    
    pnum = pNum.subs({y:i})
    display(Math('\\frac{%s}{%s} = %s' %(sym.latex(pnum),sym.latex(pDen),sym.latex(sym.simplify(pnum/pDen))) ))
    
    if sym.fraction(sym.simplify(pnum/pDen))[1]==1:
        rightnumber = i

print( 'When y=%g, there is no denominator!' %rightnumber)

```

## Factoring polynomials

```python
x,y = sym.symbols('x,y')

po = x**2 + 4*x + 3
sym.factor(po)
```

```python
# with output

fac = sym.factor(po)
print(fac)
```

```python
# not every polynomial can be factored!
po = x**2 + 4*x - 3

sym.factor(po)
```

```python
expr = 2*y**3 - 2*y**2 - 18*y + 18
sym.factor(expr)
```

```python
# multiple variables
expr = 2*x**3*y - 2*x**2*y**2 + 2*x**2*y + 6*x**2 - 6*x*y + 6*x
sym.factor(expr)

```

### Exercise

```python
# test whether factorable, print if so.

exprs = [ x**2+4*x+3 , 2*y**2-1 , 3*y**2+12*y ]

for expri in exprs:
    tmp = str( sym.factor(expri) )
    
    if tmp.find('(')!=-1:
        display(Math('%s \\quad\\Rightarrow\\quad %s' %(sym.latex(sym.expand(expri)),sym.latex(sym.factor(expri)))))
    else:
        display(Math('%s \\quad\\Rightarrow\\quad\\text{ not factorable!}' %sym.latex(sym.expand(expri))))
    
```

## Algebra 1 BUG HUNT!

```python
# from sympy.abc import x2
x2 = sym.symbols('x2')

x2 = 4
```

```python
a,b,c,d = sym.symbols('a,b,c,d')

expr = 4*b + 5*a*a - c**3 + 5*d

```

```python
import math
math.gcd(30,50)
```

```python
from sympy.abc import x

expr = 4*x - 8
sym.solve(expr)
```

```python
import numpy as np

A = np.array( [ [1,2],[3,4] ] )
# make it look nice
display(Math(sym.latex(sym.sympify(A))))
```

```python
fact_dict = sym.factorint(44)
allkeys = fact_dict.keys()

for i in fact_dict:
    print('%g was present %g times.' %(i,fact_dict[i]))

```

```python
x,y = sym.symbols('x,y')

expr = 4*x - 5*y**2

expr.subs({x:5})

```

```python
# goal is to show a fraction

f = sym.sympify(5)/9

display(Math(sym.latex(f)))

```

```python
# print the last 3 items from a list
lst = [1,3,2,5,4,6,7,5,3,7]
lst[-3:]

```

```python
from sympy.abc import x,y

expr = 2*x + 4*y

# solve for y
sym.solve(expr,y)
```

```python
import numpy as np

A = np.array( [ [1,2],[3,4] ] )

# set the element in the second row, second column to 9
A[1,1] = 9
print(A)
```
