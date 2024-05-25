Some new terminology when dealing with multiple variables
* $x_1,x_2,x_3...$ represents our features
* $x_j$ represents the list of features
* n = number of features
* $\vec{x}^{(i)}$ - features of ith training example. It will be a row vector
* $x_j^{(i)}$ = value of feature j  in ith training example

$$f_w,b(x) = w_1x_1+w_2x_2+...+w_nx_n+b$$

We can write it as 
$$\vec{w} = [w_1,w_2,w_3...w_n]$$

b is a single number
w and b are the parameters of the model
$$\vec{x} = [x_1,x_2,x_3...x_n]$$
So we can write the model as 
$$f_{\vec{w},b}(\vec{x}) = \vec{w}\cdot \vec{x} + b$$ This is called **multiple linear regression** not multivariate regression.