Cost function tells how us how well the model is doing. 
In $f_{w,b}(x) = wx+b$, w and b are called parameters. We want to find w, b: such that $\hat y^{(i)}$ is close to $y^{(i)}$  for all $(x^{(i)}, y^{(i)})$.
Our cost function can be
$$J(w,b) = \frac1{2m}\sum^m_{i=1}(\hat{y}^{(i)} - y^{(i)})^2$$
We compute the average square error as if the training set gets large the "cost" will grow larger. By convention we also divide by 2, to make our later calculations easier. This is called the squared error cost function.
We can then rewrite it as:
$$J(w,b) = \frac1{2m}\sum^m_{i=1}(f_{w,b}(x^{(i)})- y^{(i)})^2$$

Our goal is to minimize J as much as possible $$minimize_{w,b} J(w,b)$$
The cost function depends on the value of w and b, meanwhile f is a function purely on x.![[Pasted image 20240521235320.png]]
When you take into account b: we get a graph like this
![[Pasted image 20240521235609.png]]
We can plot it using a contour image, each horizontal point represents an ellipsis.
![[Pasted image 20240521235723.png]]

#### Our Cost Function in Python
```python
def compute_cost(x, y, w, b): 
    """
    Computes the cost function for linear regression.
    
    Args:
      x (ndarray (m,)): Data, m examples 
      y (ndarray (m,)): target values
      w,b (scalar)    : model parameters  
    
    Returns
        total_cost (float): The cost of using w,b as the parameters for linear regression
               to fit the data points in x and y
    """
    # number of training examples
    m = x.shape[0] 
    
    cost_sum = 0 
    for i in range(m): 
        f_wb = w * x[i] + b   
        cost = (f_wb - y[i]) ** 2  
        cost_sum = cost_sum + cost  
    total_cost = (1 / (2 * m)) * cost_sum  

    return total_cost
```


### Summary

![[Pasted image 20240521235453.png]]