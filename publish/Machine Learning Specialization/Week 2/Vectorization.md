Using vectorization makes your code shorter and allow it to run more efficiently, it allows you to take advantage of vector libraries and GPUs.

**in linear algebra: count from 1**
```python
w = np.array([1.0, 2.5, -3.3])
b = 4
x = np.array([10,20,30])
```
Without vectorization it would look like
```python
f = w[0] * x[0] + w[1] * x[i] .... +b
```

Without vectorization using a for loop
```python
f = 0
for j in range(n):
 f = f + w[j] * x[j]
f = f + b
```


With vectorization
```python
f = np.dot(w,x) + b
```
This NumPy dot function is a vectorized implementation of the dot function and when n is large it runs more faster.
It has two benefits:
* makes your code shorter
* allows your code to run much faster using parallel hardware in your computer.
![[Pasted image 20240522232330.png]]
![[Pasted image 20240522232549.png]]


