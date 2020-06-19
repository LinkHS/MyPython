# Algebra BUG HUNT!


```python
# create a list
lst = [1;3;4;1;6]

print( lst )
```


```python
# add all the numbers together
print( np.add(lst) )
```


```python
# plot the cumulative sum of a list of numbers
l = np.arange(-4,10)

plt.plot(l,'rs-')
plt.plot(np.sum(1),'bo-')
plt.legend({'list','cumsum'})
plt.show()
```


```python
### the equation:
# 4 - 2x + 5x^3

# define the coefficients
coefs = [4,-2,5]

# solve
roots = np.roots(coefs)

# and display
from sympy.abc import x

p = sym.Poly(coefs,x)

display(Math(sym.latex(p)))
```


```python
def quadeq(a,b,c):
    
    # compute
    out[0] = ( -b - sp.sqrt(b**2 - 4*a*c) ) / (2*a)
    out[1] = ( -b + sp.sqrt(b**2 - 4*a*c) ) / (2*a)
    
    # output
    return out

# test the function
print(quadeq(1,4,2))
```


```python
# create a complex number

real_part = 4
imag_part = -6

cn = np.complex(imag_part,real_part)

plt.plot(np.real(cn),np.imag(cn),'ro')
plt.grid('on')
plt.axis('square')
plt.axis([-10,10,-10,10])
plt.show()
```


```python
# symbolic complex number
a,b = sym.symbols('a,b',real=True)

z = a + b*sym.1j
display(Math('z\\times z^* = %s' %sym.latex(sym.expand(z*sym.conjugate(z)))))
```


```python
# define the phase angles
x = np.linspace(0,2*np.pi,100)

# generate the plot
plt.plot(np.cos(x),np.sin(x),'k')

# draw one vector from the origin
phs = 1*np.pi/4
plt.plot([0,np.cos(phs)],[0,np.sin(phs)],'r-')
plt.plot(np.cos(phs),np.sin(phs),'ro')


# draw axis lines
plt.plot([-1.3,1.3],[0,0],'--',color=[.8,.8,.8])
plt.plot([0,0],[-1.3,1.3],'--',color=[.8,.8,.8])

# make it look nicer
plt.axis('square')
plt.axis([-1.3,1.3,-1.3,1.3])
plt.xlabel('sine(x)')
plt.ylabel('cosine(x)')
plt.plot()

plt.show()
```


```python
a =   2 # lower bound
b = 100 # upper bound
n =   3 # number of steps

lo = np.logspace(np.log10(a),np.log10(b))
li = np.linspace(a,b)

plt.plot(li,lo,'s-',label='log')
plt.plot(li,li,'o-',label='linear')

plt.legend(['linear','log'])
plt.axis('square')
plt.show()
```


```python
## Goal is to plot the point on the function closest to f(x)=.5


# x range
x = np.linspace(-6,10,20)

# the function f(x)
fx = 1/(1+np.exp(x))

# function maximum
fmaxidx = np.argmin(abs(fx-.5))

# draw the function
plt.plot(x,fx,'bo-')
plt.plot(fmaxidx,fx[fmaxidx],'rs')

plt.show()
```


```python
## Goal is to find the local minima by adding one character

from scipy.signal import find_peaks

# x range
x = np.linspace(0,12*np.pi,200)

# the function f(x)
fx = -( np.cos(x) + x**(1/2) )

# find peaks
peeks = find_peaks(fx)

# draw the function
plt.plot(x,fx)
plt.plot(x[peeks[0]],fx[peeks[0]],'o')
plt.show()

```
