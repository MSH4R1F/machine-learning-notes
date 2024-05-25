We have a cost function $J(w,b)$, we want $min_{w,b} J(w,b)$. 
Outline:
* Start with some w,b (set w = 0, b = 0)
* Keep changing w,b to reduce J(w,b)
* Until we settle at or near a minimum
There may be more than one minimum.  

### Gradient Descent Algorithm
* $w = w-\alpha \frac{\partial}{\partial w}J(w,b)$ - adjusting w by only a bit.
	* $\alpha$ - learning rate - a positive number between 0 and 1, it controls how big of a step we take downhill.
	* $\frac{d}{dw}J(w,b)$ - derivative term of the cost function J, which direction we want to take the step and also the size.
* $b = b-\alpha \frac{\partial d}{\partial b}J(w,b)$
* Repeat until algorithm converges, we simultaneously update w and b. 

We use the partial derivative instead of the derivative.
![[Pasted image 20240522001919.png]]

### Learning Rate
The choice of alpha has a huge impact on the efficiency of our implementation on GD and it may not even work at all. 
If $\alpha$ is too small, so you take small steps to get to a minimum, gradient descent may be slow.
If $\alpha$ is too large, you may converge even away from the minimum, the gradient descent may overshoot, and never reach the minimum, it can even diverge.

If you're at a local minimum, gradient descent does nothing.


## Gradient Descent for Linear Regression

![[Pasted image 20240522002546.png]]![[Pasted image 20240522002719.png]]
A convex function has only other minima.

"Batch" gradient descent - Each step of gradient descent uses all the training examples


### Python code to compute Gradient
This python code below returns the partial derivatives of J(w,b) with respect to both  w and b.
```python
def compute_gradient(x, y, w, b): 
    """
    Computes the gradient for linear regression 
    Args:
      x (ndarray (m,)): Data, m examples 
      y (ndarray (m,)): target values
      w,b (scalar)    : model parameters  
    Returns
      dj_dw (scalar): The gradient of the cost w.r.t. the parameters w
      dj_db (scalar): The gradient of the cost w.r.t. the parameter b     
     """
    
    # Number of training examples
    m = x.shape[0]    
    dj_dw = 0
    dj_db = 0
    
    for i in range(m):  
        f_wb = w * x[i] + b 
        dj_dw_i = (f_wb - y[i]) * x[i] 
        dj_db_i = f_wb - y[i] 
        dj_db += dj_db_i
        dj_dw += dj_dw_i 
    dj_dw = dj_dw / m 
    dj_db = dj_db / m 
        
    return dj_dw, dj_db
```

### Gradient Descent Algorithm in Python
```python
def gradient_descent(x, y, w_in, b_in, alpha, num_iters, cost_function, gradient_function): 
    """
    Performs gradient descent to fit w,b. Updates w,b by taking 
    num_iters gradient steps with learning rate alpha
    
    Args:
      x (ndarray (m,))  : Data, m examples 
      y (ndarray (m,))  : target values
      w_in,b_in (scalar): initial values of model parameters  
      alpha (float):     Learning rate
      num_iters (int):   number of iterations to run gradient descent
      cost_function:     function to call to produce cost
      gradient_function: function to call to produce gradient
      
    Returns:
      w (scalar): Updated value of parameter after running gradient descent
      b (scalar): Updated value of parameter after running gradient descent
      J_history (List): History of cost values
      p_history (list): History of parameters [w,b] 
      """
    
    # An array to store cost J and w's at each iteration primarily for graphing later
    J_history = []
    p_history = []
    b = b_in
    w = w_in
    
    for i in range(num_iters):
        # Calculate the gradient and update the parameters using gradient_function
        dj_dw, dj_db = gradient_function(x, y, w , b)     

        # Update Parameters using equation (3) above
        b = b - alpha * dj_db                            
        w = w - alpha * dj_dw                            

        # Save cost J at each iteration
        if i<100000:      # prevent resource exhaustion 
            J_history.append( cost_function(x, y, w , b))
            p_history.append([w,b])
        # Print cost every at intervals 10 times or as many iterations if < 10
        if i% math.ceil(num_iters/10) == 0:
            print(f"Iteration {i:4}: Cost {J_history[-1]:0.2e} ",
                  f"dj_dw: {dj_dw: 0.3e}, dj_db: {dj_db: 0.3e}  ",
                  f"w: {w: 0.3e}, b:{b: 0.5e}")
 
    return w, b, J_history, p_history #return w and J,w history for graphing
```