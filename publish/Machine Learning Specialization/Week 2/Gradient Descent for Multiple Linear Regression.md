
Parameters: $w_1,...w_n$ and $b$ or $\vec{w}$
Model : $f_{\vec{w},b} = \vec{w}\cdot \vec{x} + b$
Cost function : $J(w_1,..,w_n,b)$  or $J(\vec{w},b)$

![[Pasted image 20240522235943.png]]![[Pasted image 20240523000057.png]]

### An alternative to gradient descent
Normal equation:
* Only for linear regression
* Solve for w, b without iterations
Disadvantages:
* Doesn't generalize to other learning algorithms
* Slow when number of features is large



### Implementing Gradient Descent for Multiple Variables

```python
import copy, math
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('./deeplearning.mplstyle')
np.set_printoptions(precision = 2) #reduced display precision on np arrays
```

```python
X_train = np.array([[2104, 5, 1, 45], [1416, 3, 2, 40], [852, 2, 1, 35]])
y_train = np.array([460, 232, 178])
```
X_train is now a matrix, where each row of the matrix represents one example. When you have m training examples and there are n features, X is a matrix with dimensions (m,n) (m rows, n columns).
So X_Shape would return (3,4) while y_shape would return (3)

w is a vector that has n elements = number of elements in one row of X.
```python
b_init = 785.1811367994083
w_init = np.array([ 0.39133535, 18.75376741, -53.36032453, -26.42131618])

```


Single prediction, vector, using the formula of the model
```python
def predict(x, w, b):
 p = np.dot(x, w) + b
 return p
 
```

![[Pasted image 20240523001451.png]]

```python
def compute_cost(X, y, w, b): 
    """
    compute cost
    Args:
      X (ndarray (m,n)): Data, m examples with n features
      y (ndarray (m,)) : target values
      w (ndarray (n,)) : model parameters  
      b (scalar)       : model parameter
      
    Returns:
      cost (scalar): cost
    """
    m = X.shape[0]
    cost = 0.0
    for i in range(m):                                
        f_wb_i = np.dot(X[i], w) + b           #(n,)(n,) = scalar (see np.dot)
        cost = cost + (f_wb_i - y[i])**2       #scalar
    cost = cost / (2 * m)                      #scalar    
    return cost
```
![[Pasted image 20240523001527.png]]

Many ways to implement gradient descent:
```python
def compute_gradient(X, y, w, b): 
    """
    Computes the gradient for linear regression 
    Args:
      X (ndarray (m,n)): Data, m examples with n features
      y (ndarray (m,)) : target values
      w (ndarray (n,)) : model parameters  
      b (scalar)       : model parameter
      
    Returns:
      dj_dw (ndarray (n,)): The gradient of the cost w.r.t. the parameters w. 
      dj_db (scalar):       The gradient of the cost w.r.t. the parameter b. 
    """
	m,n = X.shape
	dj_dw = np.zeroes((n,))
	dj_db = 0
	for i in range(m):
	  err = (np.dot(X[i], w) + b) - y[i]
	  for j in range(n):
		  dj_dw[j] = dj_dw[j] + err * X[i,j]
	  dj_db = dj_db + err
	dj_dw = dj_dw / m
	dj_db = dj_db / m
	return dj_db, dj_dw
	  	  
```

```python
def gradient_descent(X, y, w_in, b_in, cost_function, gradient_function, alpha, num_iters): 
    """
    Performs batch gradient descent to learn w and b. Updates w and b by taking 
    num_iters gradient steps with learning rate alpha
    
    Args:
      X (ndarray (m,n))   : Data, m examples with n features
      y (ndarray (m,))    : target values
      w_in (ndarray (n,)) : initial model parameters  
      b_in (scalar)       : initial model parameter
      cost_function       : function to compute cost
      gradient_function   : function to compute the gradient
      alpha (float)       : Learning rate
      num_iters (int)     : number of iterations to run gradient descent
      
    Returns:
      w (ndarray (n,)) : Updated values of parameters 
      b (scalar)       : Updated value of parameter 
      """
	 # An array to store cost J and w's at each iteration primarily for graphing later
    J_history = []
    w = copy.deepcopy(w_in)  #avoid modifying global w within function
    b = b_in
    
    for i in range(num_iters):

        # Calculate the gradient and update the parameters
        dj_db,dj_dw = gradient_function(X, y, w, b)   ##None

        # Update Parameters using w, b, alpha and gradient
        w = w - alpha * dj_dw               ##None
        b = b - alpha * dj_db               ##None
      
        # Save cost J at each iteration
        if i<100000:      # prevent resource exhaustion 
            J_history.append( cost_function(X, y, w, b))

        # Print cost every at intervals 10 times or as many iterations if < 10
        if i% math.ceil(num_iters / 10) == 0:
            print(f"Iteration {i:4d}: Cost {J_history[-1]:8.2f}   ")
        
    return w, b, J_history #return final w,b and J history for graphing

```