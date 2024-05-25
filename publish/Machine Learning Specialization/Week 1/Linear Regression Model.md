Linear Regression - Fitting a straight line to your data.
![[Pasted image 20240521133954.png]]
It is an example of supervised learning. It is called a regression model as it predicts numbers.
Terminology:
* **Training Set**: Data used to train the model, our "points"
* x = "input" variable feature
* y = "output" ("target") variable
* m = number of training examples
* $(x^{(i)}, y^{(i)})$ = ith training example


To train the model, you feed both the input features and output features into a supervised learning algorithm, which produces a function (hypothesis) $f$. $f$ takes in an input x and outputs a prediction $\hat{y}$ .  The function f is called the model, x is the feature and y is the prediction.

$f_{w,b}(x) = wx+b$ , this is a linear regression with a single feature x (one variable), or otherwise known as **univariate** linear regression.


### Plotting a Linear Regression using Python
We import both NumPy (used for scientific computing) and Matplotlib (for plotting data).

```python
import numpy as np
import matplotlib.pyplot as plt
pl.style.use('./deeplearning.mplstyle')
```

We can initialize our training data using np.array
```python
x_train = np.array(...)
y_train = np.array(...)
# m is the number of training examples
m = x_train.shape[0] or m = len(x_train)
```

We can plot the two variables using scatter() function in matplotlib, the function arguments `marker` and `c` show the points as red crosses.
```python
plt.scatter(x_train, y_train, marker = 'x', c ='r')
plt.title(...)
plt.ylabel(...)
plt.xlabel(...)
plt.show()
```