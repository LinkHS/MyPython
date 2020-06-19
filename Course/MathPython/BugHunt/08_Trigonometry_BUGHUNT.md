```python
import numpy as np
import matplotlib.pyplot as plt
```

# Trigonometry BUG HUNT!

```python
# plot a series of random numbers
s = np.randn(100,1)
plt.imshow(s)
plt.show()
```

```python
# create and image a matrix of random integers between (and including) 3 and 20

mat = np.random.randint(3,20,30,20)

plt.imshow(mat)
plt.colorbar()
plt.show()
```

```python
# create 100 random phase angles [0,2pi] and show unit vectors with those angles

n = 100

randphases = np.random.rand(n)+2*pi

for i in range(0,n):
    plt.polar(randphases[i],1,'o',color=np.random.rand(3),markersize=20,alpha=.3)
    
plt.show()

```

```python
# create an outwards spiral using phase angles and amplitudes

n = 100
a = np.linspace(0,1,n)
p = np.linspace(0,4*np.pi,n)

plt.plot(p,A);
```

```python
# convert radians to degrees

n = 10
rad = np.logspace(np.log10(1),np.log10(360),n)

print(np.rad2deg(rad))
```

```python
# famous equality in trigonometry

ang = np.logspace(np.log10(0),np.log10(2*np.pi),10)
np.cos(ang)*2 + np.sin(ang)*2
```

```python
# create euler's number
p = np.pi/4
m = .5

eulr = 1j*np.exp(m*p)

# now extract magnitude and phase
mag = np.abs(eulr)
ang = np.angle(eulr)

# then plot
plt.polar([0,mag],[0,ang],'b',linewidth=3)
plt.show()
```
