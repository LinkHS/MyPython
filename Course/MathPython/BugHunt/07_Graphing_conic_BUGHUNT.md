```python
import numpy as np
import matplotlib.pyplot as plt
```

# Conics BUG HUNT!

```python
# Make a gaussian
x = np.linspace(-2,2,100)

# create the gaussian
X,Y = np.meshgrid(x,x)
gauss2d = np.exp( -(X**2+Y)**2 )

# and plot
plt.imshow(gauss2d)
plt.axis('off')
plt.show()
```

```python
# draw a circle using meshgrid
r = 3

# grid space
x = np.linspace(-r,r,100)
X,Y = np.meshgrid(x)

# create the function
Fxy = X**2 + Y**2 - r**2

# draw it
plt.imshow(Fxy)

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
Fxy = X**2/a**2 + Y**2/b**2

# draw it as a contour
plt.contour(X,Y,Fxy,0)

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
Fxy = (X-h)**2/a**2 - (Y-k)**2/b**2

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
