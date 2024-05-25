To make sure GD is working correctly, we can plot the cost function for every iteration of GD. This is called the learning curve. If it is working then J should decrease after every iteration, if it increases then our alpha could be too large or there's a bug in the code
![[Pasted image 20240523005244.png]]
We can use an automatic convergence test as well.
Let $\epsilon$ be $10^{-3}$, if J(w,b) decreases by <= $\epsilon$ in one iteration, declare convergence

### Choosing a good alpha

![[Pasted image 20240523005633.png]]
With a small enough $\alpha$ J(w,b) should decrease on every iteration. If alpha is too small, then gradient descent would take a long time to converge.


