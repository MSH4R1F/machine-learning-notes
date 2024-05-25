## Supervised Learning
Supervised Learning refers to algorithms that learn input to output mappings. You give your learning algorithm examples to learn from. The algorithm eventually learn to take the input label and predict at a high accuracy the output label.

X - > Y
email -> spam (0,1)  = spam filtering
audio -> text transcripts = speech recognition
English -> Spanish = machine translation
ad, user info -> click ad? (0,1) = online advertising
image, radar info -> position of other cars = self-driving cars


### Regression Lines
One simple way we do this is using regression lines. Whether we use a straight line, or a curve is explained further on on the course.
Regression - Predict a number from infinitely many outputs.
![[Pasted image 20240520143921.png]]


### Classification

> Example: Breast Cancer Detection
> We are developing an algorithm that can predict depending on the tumour size if the tumour is malignant or benign.

Here we are trying to predict one of the two categories: (0) for benign and (1) for malignant. This differs from regression which tries to predict a number from an infinite number of outputs. This is why we call this type of algorithm classification. 
![[Pasted image 20240520144759.png]]
In classification problems you can have more than 2 possible categories. The term **output classes** and **output categories** are used interchangeably.
It is possible for inputs to be more than one dimension, often in many problems we feed in a large number of inputs.
![[Pasted image 20240520144953.png]]

Summary: Classification algorithms try to predict categories, it predicts a small number of possible outputs.

