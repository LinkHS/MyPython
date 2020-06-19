# Linear algebra BUG HUNT!!

```python
# create a column vector
cv = np.array([ (-2, 3 )])

display(Math(sym.latex(sym.sympify(cv))))
```

```python
# visualize scalar-vector multiplication

v = np.array([-2,2])
s = .7

sv = np.array([s,s]).T@v
plt.plot([0,v[0]],[0,v[1]],'ro-',linewidth=3,label='v')
plt.plot([0,sv],[0,sv],'o-',linewidth=3,label='%sv')
    
plt.axis = 'square'
plt.legend()
plt.axis([-3,3,-3,3])
plt.grid()
plt.show()
```

```python
# algorithm to compute the dot product
v = np.random.randn(7)
w = np.random.randn(8)

dp1 = 0
for i in range(0,len(v)):
    dp1 = dp1 + v[i]*w[1]

dp2 = np.dot(v,w)

print(str(dp1) + '\t' + str(dp2))
```

```python
# number of data points
n = 10

# data
data1 = np.arange(0,n) + np.random.randn(n)
data2 = np.arange(0,n) + np.random.randn(n)

# compute correlation
numer = np.dot(data1,data2)
denom = np.sqrt( np.dot(data1,data1) ) * np.sqrt(np.dot(data2,data2))
r1 = numer/denom

# confirm with numpy function
r2 = np.corrcoef(data1,data2)[1][0]

print(r1)
print(r2)
```

```python
# outer product computation
o1 = np.random.randint(0,10,7)
o2 = np.random.randint(0,10,4)

outermat = np.zeros((len(o1),len(o2)))

for i in range(len(o2)):
    outermat[i,:] = o1*o2[i]
    
print(outermat-np.outer(o1,o2))
```

```python
# matrix multiplication
A = np.random.randn(5,5)
I = np.eye(5)

A*I
```

```python
# matrix multiplication
A = np.random.randn(8,5)
I = np.eye(5)

print(A)
print(' ')
print(A*I)
```

```python
# random matrices are invertible
A = np.random.randint(-5,6,(5,5))
Ainv = np.inv(A)

np.round(A@Ainv,4)
```

```python
# plot the eigenspectrum
# the matrix
M = np.random.randint(-5,5,(5,5))
M = M@M.T

# its eigendecomposition
eigvecs,eigvals = np.linalg.eig(M)

plt.plot(np.matrix.flatten(eigvals),'s-')
plt.xlabel('Components')
plt.ylabel('Eigenvalues')

plt.show()
```

```python
# Reconstruct a matrix based on its SVD
A = np.random.randint(-10,11,(10,20))

U,s,V = np.linalg.svd(A)

# reconstruct S
S = np.diag(s)

Arecon = U@V@S

fig,ax = plt.subplots(1,3)

ax[0].imshow(A,vmin=-10,vmax=10)
ax[0].set_title('A')

ax[1].imshow(Arecon,vmin=-10,vmax=10)
ax[1].set_title('Arecon')

ax[2].imshow(A-Arecon,vmin=-10,vmax=10)
ax[2].set_title('A-Arecon')

plt.show()
```
