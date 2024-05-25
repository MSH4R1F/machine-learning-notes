Feature 
![[Pasted image 20240523004412.png]]
When you have different features that take on very different ranges of values, it can cause gradient descent to run slowly but rescaling the different features so they all take on comparable range of values, can speed up gradient descent significantly.
We want the contour lines to be more circular and less oval.

![[Pasted image 20240523004710.png]]
Dividing by the maximum

In addition to dividing by the maximum we can also centre them around the mean.
![[Pasted image 20240523004846.png]]

Z-score normalization - we calculate the standard deviation, we calculate the mean.
![[Pasted image 20240523004946.png]]
Feature scaling:
* aim for about -1 <= x_j <= 1 for each feature $x_j$, but its not a strict limit.![[Pasted image 20240523005109.png]]

```python
def zscore_normalize_features(X):
    """
    computes  X, zcore normalized by column
    
    Args:
      X (ndarray (m,n))     : input data, m examples, n features
      
    Returns:
      X_norm (ndarray (m,n)): input normalized by column
      mu (ndarray (n,))     : mean of each feature
      sigma (ndarray (n,))  : standard deviation of each feature
    """
    # find the mean of each column/feature
    mu     = np.mean(X, axis=0)                 # mu will have shape (n,)
    # find the standard deviation of each column/feature
    sigma  = np.std(X, axis=0)                  # sigma will have shape (n,)
    # element-wise, subtract mu for that column from each example, divide by std for that column
    X_norm = (X - mu) / sigma      

    return (X_norm, mu, sigma)
 
#check our work
#from sklearn.preprocessing import scale
#scale(X_orig, axis=0, with_mean=True, with_std=True, copy=True)
```