# Graphing bug hunt!


```python
plt.plot(3,2,'ro')

# set axis limits
plt.axis('square')
plt.axis(-6,6,-6,6)
plt.show()
```


```python
# plot a line
plt.plot([0 3],[0 5])
plt.show()
```


```python
import numpy as np

x = range(-3,4)
y = np.zeros(len(x))

for i in range(0,len(x)):
    y[i] = 2 - x**2

plt.plot(x,y,'s-')
plt.show()
```


```python
# plot two lines
plt.plot([-2,3],[4,0],'b')
plt.plot([0,3],[-3,3],'r')

plt.legend()
plt.show()
```


```python
randmat = np.random.randn(5,9)

# draw a line from lower-left corner to upper-right corner
plt.plot([0,4],[8,0],color=(.4,.1,.9),line_width=5)

plt.imshow(randmat)
plt.set_cmap('Purples')
plt.show()
```


```python
# plot two lines
plt.plot([-2,3],[4,0],'b',label='line1')
plt.plot([0,3],[-3,3],'r',label='line2')

plt.legend('on')
plt.show()
```


```python
x = np.linspace(1,4,20)
y = x**2/(x-2)

plt.plot(x,y)

# adjust the x-axis limits according to the first and last points in x
plt.xlim(x[0,-1])

plt.show()
```


```python
x = sym.symbols('x')
y = x**2 - 3*x

xrange = range(-10,11)

for i in range(0,len(xrange)):
    plt.plot(xrange[i],y(xrange[i]),'o')
        
plt.xlabel('x')
plt.ylabel('$f(x) = %s$' %sym.latex(y))
plt.show()
```


```python
x = [-5,5]
m = 2
b = 1

y = m*x+b

plt.plot(x,y)
plt.show()
```


```python
x = range(-20,21)

for i in range(0,len(x)):
    plt.plot([0,x[i]],[0,abs(x[i])**(1/2)],line_color=(i/len(x),i/len(x),i/len(x))
    
plt.axis('of')
plt.show()
```


```python
# draw a checkerboard with purple numbers on top

m = 8
n = 4

# initialize matrix
C = np.zeros((m,4))

# populate the matrix
for i in range(0,m):
    for j in range(0,n):
        C[i,j] = (-1)**(i+j)
        

# display some numbers
for i in range(0,m):
    for j in range(0,n):
        plt.text(i,j,i+j,\
                 horizontalalignment='center',verticalalignment='center',\
                 fontdict=dict(color='m'))


plt.imshow(C)
plt.set_cmap('gray')
plt.show()
```
