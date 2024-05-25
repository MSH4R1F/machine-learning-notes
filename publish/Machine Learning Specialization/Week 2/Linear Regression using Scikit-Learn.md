An open-source, commercially usable ML toolkit called scikit-learn contains implementations of many of the algorithms we work with.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
from lab_utils_multi import  load_house_data
from lab_utils_common import dlc
np.set_printoptions(precision=2)
plt.style.use('./deeplearning.mplstyle')
```
Scikit-learn has a gradient descent regression model called `sklearn.linear_model.SGDRegressor`. This model performs best with normalized inputs. `sklearn.preprocessing.StandardScaler` will perform z-score normalization.

### Load the data set
```python
X_train, y_train = load_house_data()
X_features = ['size(sqft)', 'bedrooms', 'floors', 'age']
```

### Scale/Normalize the data
```python
scaler = StandarScaler()
X_norm = scaler.fit_transform(X_train)
```

### Create and fit the regression Model
```python
sgdr = SGDRegressor(max_iter=1000)
sgdr.fit(X_norm, y_train)
```

### View parameters
```python
b_norm = sgdr.intercept_
w_norm = sgdr.coef_
```

### Make predictions
```python
# predict using sgdr.predict()
y_pred_sgd = sgdr.predict(X_norm)
# Predict manually using w,b
y_pred = np.dot(X_norm, w_norm) + b_norm
```