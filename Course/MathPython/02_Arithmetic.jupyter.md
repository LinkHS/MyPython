# Arithmetic


## Addition, subtraction, multiplication, division


### Addition

```python
# use Python like a calculator
4+5 # press Ctrl-Enter to find the answer!
```

```python
# can use multiple numbers at the same time:
4 + 3/4 + 9.4
```

### Subtraction

```python
# exactly how you'd think it should work!
4 - 3 - 1
```

### Multiplication

```python
# Use * to indicate multiplication (not x!)
3*4 
```

```python
# can mix operations
3*4 - 3*2
```

### Division

```python
# use the forward slash
3/4
```

### Exercises!

```python
#1)
5 - (2/4) * (3/5)
```

```python
#2)
(4-5) / 3+5*6
```

## Using variables in place of numbers

```python
# create a variable by assigning a number to it
x = 17
```

```python
# then use that variable in place of a number
x - 2
```

```python
# you can re-assign variables whenever you need to
x = 10
# note that nothing prints out to the screen here. 
# We didn't ask Python to compute anything, just to set a variable.
```

```python
# you can even use a variable to reassign itself!
x = x-2*5
x
```

```python
# multiple variables can be really useful
x = 10
y = 20

try:
    # can't use a variable before it's created!
    x + y*z
except Exception as E:
    print("Exception: {}".format(type(E).__name__))
    print("Exception message: {}".format(E))
```

```python
z = 30
x + y*z
```

```python
# only the final line of code will print out an answer;
# otherwise, you can use the print() function

40+10

50+3

print(40+3)
print(10+6)
```

### Exercises

```python
# 1)
x = 7
y = -2
z = 5

# 1)
3*x*(4+y)
```

```python
# 2)
-y - (x+3)/z
```

## Printing out equations in IPython

```python
# the line below will import special functions that are not otherwise available in the main Python environment
from IPython.display import display, Math

display('4 + 3 = 7')

display('4 + 3 = ' + str(4+3))

```

```python
# can also use string variables for printing

expr = '4+5-9 = 0'

display(expr)
display(Math(expr)) # note that this line produces slightly nicer output
```

```python
# mixing variable types in outputs
x = 4
y = 5

display(Math( str(x) + ' + ' + str(y) + ' = ' + str(x+y) )) # note: plus sign here is for concatenating strings, not for summing!
```

```python
# without using latex
display(Math('4/5 = .8'))

# with latex (much nicer!)
display(Math('\\frac{4}{5} = ' + str(4/5)))

```

```python
# alternative method for integrating numbers with string text
display(Math('\\frac{4}{5} = %g' %(4/5)))

# Note: We'll be using the basics of substitution (%g and %s) in this course; for more details and formatting options, see
# https://docs.python.org/3/library/stdtypes.html#printf-style-string-formatting

```

```python
# more variables in the same line

x = 3.4
y = 17

display(Math('%g \\times %g = %g' %(x,y,x*y)))

```

### Exercises

```python
x = 7
y = -2
z = 5

# 1)
ans = 3*x*(4+y)
display(Math('3x(4+y) = ' + str(ans)))
display(Math('3\\times%g(4+%g) = %g' %(x,y,ans) ))


# 2)
ans = -y - (x+3)/z
display(Math('-y - \\frac{x+3}{z} = %g'%ans))

display(Math('-%g - \\frac{%g+3}{%g} = %g'%(y,x,z,ans)))
```

## Exponents (powers)

```python
# use **
3**2
```

```python
# careful not to confuse with ^
3^1
# this is actually the symbol for the XOR logical problem
```

```python
# variables can also be useful
x = 2
3**x
```

```python
# Note about importing functions or modules: You need to do this only once per session.
# Thus, you don't need the following line if you've already imported these functions.
# On the other hand, it doesn't hurt to import multiple times ;)
from IPython.display import display, Math


# using the print function
print( 2**x )

# print out the whole equation
display(Math('x^%g = %g'%(x,2**x)))
```

```python
# the law of exponents
print(3**3)
print(3*3*3)
```

```python
print( 3**2 * 3**4 )
print( 3**(2+4) )
```

```python
# square root as fractional power
x**(1/2)
```

### Exercises

```python
x = 5
y = 5.1

# 1)
ans = x**(3/4) * 4**y
display(Math('x^{3/4}\\times 4^y = %g'%ans))

# 2)
solution = (3**3) / (x**y)
display(Math('\\frac{3^3}{x^y} = %g'%solution))

# 3)
ans = 10**(x-4)
display(Math('10^{x-4} = %g'%ans))
```

## Using for-loops to compute powers

```python
# basic for loop construction

for i in [0,1,2,3]:
    print(i)

```

```python
# using the range function
list( range(0,3) )
```

```python
for i in range(0,3):
    print('This is the %gth element.'%i)
```

```python
for i in range(0,5):
    num = i**2
    print(num)
```

### Exercises

```python
# write code to print the following:
# 2 to the power of 0 is 1
# and so on for 2**0 through 2**9

base = 2

for i in range(0,10):
    print('%g to the power of %g is %g'%(base,i,base**i))
```

## Order of operations


## Reminder about order of operations:
PEMDAS:

    Parentheses
    Exponents
    Multiplication
    Division
    Addition
    Subtraction
    
(Please excuse my dear aunt Sally)


```python
# examples from slides

print( 4 + 3 * 2 - 1 ) # spacing has no effect

```

```python
# will this be 24 or 10?
4 + 2 * 3
```

```python
# how about this one? can you make the be 7 or -11?
2 - 4 + 9
```

### Exercises

```python
# add parentheses as needed to make the following equations true

# 1) 
print( 4*5/(7+3) )

# 2)
print( 9/(3+6)-1 )
```

## Testing inequalities and Boolean data type

```python
# The result of the below expression can only be True or False
4>5
```

```python
print(5<5.1)
```

```python
b = 10 > 3*3.33334

print(b)
print(type(b)) # the variable class is boolean
```

```python
4 >= 2+2
```

```python
5 <= 3*2-1.00000000001
```

```python
# testing for equalities requires two equals signs

# 4 = 4
4 == 4

test = 4 == 5
print(test)

```

### Exercises

```python
# which variable value(s) will make the following inequalities true?
# comment out the incorrect answers!
x = 2
x = 3
x = 4

# 1)
print( 4*x+3 < 17-x**2 )

#2) 
print( 8*x - 2 <= -3*x + 42 )
```

## If-statements and logical operators

```python
# start with basics:
if 4+2 == 6:
    print('True!')
else:
    print('Nope!')
```

```python
# using variables

x1 = 3
x2 = 4

if x1>x2:
    print('%g is greater than %g' %(x1,x2))
elif x1<x2:
    print('%g is greater than %g' %(x2,x1))
else:
    print('%g is equal to %g' %(x1,x2))

```

```python
# with logical operators

x = 1
y = 2
z = 3

if x+y<z or x-z>y:
    print('Yeah, the first thing.')
elif 2*x+y>z and 2*z-x>y:
    print('the second thing')

```

### Exercise

```python
from IPython.display import display, Math

for i in range(0,4):
    for j in range(0,5):
        if j>0 and i>0:
            display(Math('%g^{-%g} = %g' %(i,j,i**-j)))
```

## Absolute value

```python
a = -4
abs(a)
```

```python
b = abs(a)
a,b
```

```python
# printing out with latex
x = -4.3

# If the following line crashes, you need to import the display and Math functions as shown earlier
display(Math('|%g| = %g' %(x,abs(x)) ))
```

### Exercise

```python
# report the absolute value of a set of numbers only if they are less than -5 or greater than +2
numbers = [-4,6,-1,43,-18,2,0]

# for-loop
for i in numbers:
    if i<-5 or i>2:
        print( 'Absolute value of %g is %g.' %(i,abs(i)) )
    else:
        print( str(i) + ' was not tested.' )
```

## Remainder after division (modulus)

```python
# pick some numbers
a = 10
b = 3

# division
a/b
```

```python
# integer division
int(a/b)
```

```python
# with a remainder!
a%b
```

```python
# now for a nicer printing
# set variables for outputs
divis = int(a/b)
remainder = a%b

print("%g goes into %g, %g times with a remainder of %g." %(b,a,divis,remainder))
```

### Exercises

```python
# determining whether a number is odd or even

nums = range(-5,6)

for i in nums:
    
    # check for first character spacing
    firstchar = '' if i<0 else ' '
    
    # test and report
    if i%2 == 0:
        print('%s%g is an even number' %(firstchar,i))
    else:
        print('%s%g is an odd  number' %(firstchar,i))
```

## Create interactive math functions

```python
# create a function that takes two numbers and reports the numbers, their integer division, and remainder

# define the function
def computeremainder(x,y):
    division  = int( x/y )
    remainder = x%y
    
    print("%g goes into %g, %g times with a remainder of %g." %(y,x,division,remainder))
```

```python
computeremainder(10,3)
```

```python
# create a function that inputs two numbers and reports the numbers, their integer division, and remainder

# define the function
def computeremainder():
    x = int( input('Input numerator: ') )
    y = int( input('Input denominator: ') )
    division  = int( x/y )
    remainder = x%y
    
    print("%g goes into %g, %g times with a remainder of %g." %(y,x,division,remainder))
```

```python
# create a function that takes two numbers as input, 
# and returns the larger-magnitude number to the power of the smaller-magnitude number

def powerfun():
    n1 = int( input('Input one number ') )
    n2 = int( input('Input a second number ') )
    
    # test
    if abs(n1)>abs(n2):
        return n1,n2,n1**n2
    else:
        return n2,n1,n2**n1

```

### Exercises

```python
# input three numbers from user, x, y, and a switch option
#    Create two functions that will compute x**y and x/y
#    call the appropriate function and print the result

from IPython.display import display,Math

# create functions
def powers(x,y):
    display(Math('%g^{%g} = %g' %(x,y,x**y)))

def division(x,y):
    display(Math('\\frac{%g}{%g} = %g' %(x,y,x/y)))
    
def mainfunction():
    x = int(input('input X: '))
    y = int(input('input Y: '))
    display(Math('Press "1" to compute %g^{%g}\\text{ or "2" to compute }\\frac{%g}{%g}' %(x,y,x,y)))
    which = int(input(' '))
    
    if which==1:
        powers(x,y)
    elif which==2:
        division(x,y)
    else:
        print('Invalid selection!')

```

```script magic_args="false --no-raise-error"

# run the main function
mainfunction()
```

## Interactive functions: Guess the number!

```python
# guess a number!
# write code to generate a random integer and then input user guesses of that integer. 
# The code should tell the user to guess higher or lower until they get the right answer.

from numpy import random

def guessTheNumber():
    num2guess = random.randint(1,101)
    
    userguess = int( input('Guess a number between 1 and 100  ') )
    while userguess!=num2guess:
        if userguess>num2guess:
            print('Guess lower!')
        elif userguess<num2guess:
            print('Guess higher!')
        
        userguess = int( input('Guess again ') )
    
    print('Got it! The right number was %g and your final guess was %g'%(num2guess,userguess))
```

## ARITHMETIC BUG HUNT!!!

```python
x = '1'
y = '2'

print( int(x) + int(y) )
```

```python
display(Math('\\frac{1}{2}'))
```

```python
x = 4
y = 5
display(Math('%g + %g = %g'%(x,y,x+y)))
```

```python
display(Math('\\frac{%g}{%g} = %g'%(x,y,x/y)))
```

```python
3**2 # should equal 9
```

```python
for i in range(0,3):
    display(Math('%g\\times2 = %g'%(i,i*2)))
```

```python
a = 10
b = 20

result = 2*a <= b

print(result)
```

```python
if a+b*2 > 40:
    print(str(a+b*2) + ' is greater than ' + str(40) + '. Time to celebrate!')

```

```python
# division
4/10
```

```python
display(Math('\\frac{9}{3} = ' + str(9/3)))
```

```python
display(Math('\\frac{9}{3} = ' + str(9/3)))
```

```python
# print the numbers 1 through 10
t = 1
while t<11:
    print(t)
    t=t+1
```
