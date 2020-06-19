# Grahping conic

```python
import numpy as np
import sympy as sym
import matplotlib.pyplot as plt
```

## Graphing parabolas

```python
# parameters
n = 100
a = 1
h = 1
k = -2

# x-axis points to evaluate the function
x = np.linspace(-2,4,n)

# create the function
y = a*(x-h)**2 + k

# and plot it!
plt.plot(x,y)
plt.grid()
plt.axis('square')
plt.show()

```

```python
# parameters
n = 100
a = 1
h = 1
k = -2

# x-axis points to evaluate the function
x = np.linspace(-2,4,n)

# create the function
y = a*(x-h)**2 + k

# and plot it!
plt.plot(y,x)

plt.grid()
plt.axis('square')
plt.show()

```

### Exercise

```python
# draw the vertex, focus, and directrix

x = np.linspace(-6,6,40)
y = (x-2)**2/2 + 1

# 1) convert to standard parabola equation
# 4*p*(y-k) = (x-h)**2
# 2) write standard equation
h = 2
k = 1
p = 1/2 # 2=4p


# plot the parabola
plt.plot(x,y,label='Parabola')

# plot the vertex
plt.plot(h,k,'ro',label='Vertex')

# plot the focus
plt.plot(h,k+p,'go',label='focus')

# plot the directrix
d = k-p
plt.plot(x[[0,-1]],[d,d],label='directrix')

plt.legend()
plt.axis('square')
plt.axis([x[0],x[-1],d-.5,10])
plt.grid()
plt.show()
```

## Creating contours from meshes in Python

```python
X,Y = np.meshgrid(range(0,10),range(0,15))

plt.subplot(121)
plt.pcolormesh(X,edgecolors='k',linewidth=.1)
# plt.gca().set_aspect('equal')
plt.title('X')

plt.subplot(122)
plt.pcolormesh(Y,edgecolors='k',linewidth=.1)
plt.gca().set_aspect('equal')
plt.title('Y')

plt.show()
```

```python
x = np.linspace(0,2*np.pi,50)
y = np.linspace(0,4*np.pi,50)

X,Y = np.meshgrid(x,y)

F = np.cos(X) + np.sin(Y)
plt.imshow(F,extent=[x[0],x[-1],y[0],y[-1]])
plt.show()
```

### Exercise

```python
# Make a gaussian
x = np.linspace(-2,2,100)
s = 2

# create the gaussian
X,Y = np.meshgrid(x,x)
gauss2d = np.exp( -(X**2+Y**2)/s )

# and plot
plt.imshow(gauss2d,extent=[x[0],x[-1],x[0],x[-1]])
plt.axis('off')
plt.show()
```

## Graphing circles

```python
# circle parameters
a = 2
b = -3
r = 3

# grid space
axlim = r + np.max((abs(a),abs(b)))
x = np.linspace(-axlim,axlim,100)
y = np.linspace(-axlim,axlim,100)
X,Y = np.meshgrid(x,y)

# create the function
Fxy = (X-a)**2 + (Y-b)**2 - r**2

# draw it as a contour
plt.contour(X,Y,Fxy,0)

# draw a dot in the center
plt.plot(a,b,'go')

# draw guide lines
plt.plot([-axlim,axlim],[0,0],'k--')
plt.plot([0,0],[-axlim,axlim],'k--')

plt.gca().set_aspect('equal')
plt.show()
```

### Exercise

```python
# circle parameters
a = [-1.5,1.5]

# grid space
axlim = 5
x = np.linspace(-axlim,axlim,100)
y = np.linspace(-axlim,axlim,100)
X,Y = np.meshgrid(x,y)

for r in np.linspace(.5,3,15):
    for ai in a:
        Fxy = (X-ai)**2 + Y**2 - r**2
        plt.contour(X,Y,Fxy,0,colors=[(r/3,r/3,r/3)])
    
plt.gca().set_aspect('equal')
plt.plot(a,[0,0],'k',linewidth=3)
plt.axis('off')
plt.show()
```

## Graphing ellipses

```python
# parameters
a = 2
b = 3
h = 1
k = 2

# grid space
axlim = np.max((a,b)) + np.max((abs(h),abs(k)))
x = np.linspace(-axlim,axlim,100)
y = np.linspace(-axlim,axlim,100)
X,Y = np.meshgrid(x,y)

# create the function
Fxy = (X-h)**2/a**2 + (Y-k)**2/b**2 - 1

# draw it as a contour
plt.contour(X,Y,Fxy,0)

# draw a dot in the center
plt.plot(h,k,'go')

# draw guide lines
plt.plot([-axlim,axlim],[0,0],'--',color=[.8,.8,.8])
plt.plot([0,0],[-axlim,axlim],'k--',color=[.8,.8,.8]) # color overwrites k

plt.gca().set_aspect('equal')
plt.show()
```

### Exercise

```python
# parameters
n = 16
a = abs(np.linspace(4,-4,n))
b = 4
h = 0
k = np.linspace(-4,4,n)

# grid space
axlim = 8
x = np.linspace(-axlim,axlim,100)
y = np.linspace(-axlim,axlim,100)
X,Y = np.meshgrid(x,y)

for i in range(0,n):
    Fxy = (X-h)**2/a[i]**2 + (Y-k[i])**2/b**2 - 1
    plt.contour(X,Y,Fxy,0,colors=[(i/n,0,i/n)])


plt.gca().set_aspect('equal')
plt.axis('off')
plt.show()
```

## Graphing hyperbolas

```python
# parameters
a = 1
b = .5
h = 1
k = 2

# grid space
axlim = 2* (np.max((a,b)) + np.max((abs(h),abs(k))))
x = np.linspace(-axlim,axlim,100)
X,Y = np.meshgrid(x,x)

# create the function
Fxy = (X-h)**2/a**2 - (Y-k)**2/b**2 - 1

# draw it as a contour
plt.contour(X,Y,Fxy,0)

# draw a dot in the center
plt.plot(h,k,'go')

# draw guide lines
plt.plot([-axlim,axlim],[0,0],'--',color=[.8,.8,.8])
plt.plot([0,0],[-axlim,axlim],'k--',color=[.8,.8,.8]) # color overwrites k

plt.gca().set_aspect('equal')
plt.show()
```

### Exercise

```python
# parameters
n = 16
a = np.linspace(1,5,n)
b = np.linspace(1,5,n)

# grid space
axlim = 8
x = np.linspace(-axlim,axlim,100)
y = np.linspace(-axlim,axlim,100)
X,Y = np.meshgrid(x,y)

for i in range(0,n):
    Fxy = X**2/a[i]**2 - Y**2/b[i]**2 - 1
    plt.contour(X,Y,Fxy,0,colors=[(i/n,0,i/n)])

    Fxy = -X**2/a[i]**2 + Y**2/b[i]**2 - 1
    plt.contour(X,Y,Fxy,0,colors=[(0,i/n,i/n)])

plt.gca().set_aspect('equal')
plt.axis('off')
plt.show()
```

## Conics BUG HUNT!

```python
# Make a gaussian
x = np.linspace(-2,2,100)

# create the gaussian
X,Y = np.meshgrid(x,x)
gauss2d = np.exp( -(X**2+Y**2) )

# and plot
plt.imshow(gauss2d)
plt.axis('off')
plt.show()
```

```python
# draw a circle using meshgrid
r = 2

# grid space
x = np.linspace(-r,r,100)
y = np.linspace(-r,r,100)
X,Y = np.meshgrid(x,y)

# create the function
Fxy = X**2 + Y**2 - r**2

# draw it
plt.imshow(Fxy)
plt.contour(Fxy,0,colors='k')

plt.axis('off')
plt.show()
```

```python
# parameters
a = 1
b = 2
h = 2
k = -3

# grid space
axlim = np.max((a,b)) + np.max((abs(h),abs(k)))
x = np.linspace(-axlim,axlim,100)
y = np.linspace(-axlim,axlim,100)
X,Y = np.meshgrid(x,y)

# create the function
Fxy = (X-h)**2/a**2 + (Y-k)**2/b**2 - 1

# draw it as a contour
plt.contour(X,Y,Fxy,0)
plt.plot(h,k,'go')

# draw guide lines
plt.grid()
plt.title('Ellipse centered at (x,y)=(%s,%s)' %(h,k))
plt.gca().set_aspect('equal')
plt.show()
```

```python
# hyperbola! (not an "X")

# parameters
a = 1
b = .5
h = 1
k = 2

# grid space
axlim = 2* (np.max((a,b)) + np.max((abs(h),abs(k))))
x = np.linspace(-axlim,axlim,100)
y = np.linspace(-axlim,axlim,100)
X,Y = np.meshgrid(x,y)

# create the function
Fxy = (X-h)**2/a**2 - (Y-k)**2/b**2 - 1

# draw it as a contour
plt.contour(X,Y,Fxy,0)

# draw a dot in the center
plt.plot(h,k,'go')

# draw guide lines
plt.plot([-axlim,axlim],[0,0],'--',color=[.8,.8,.8])
plt.plot([0,0],[-axlim,axlim],'--',color=[.8,.8,.8])

plt.gca().set_aspect('equal')
plt.show()
```
