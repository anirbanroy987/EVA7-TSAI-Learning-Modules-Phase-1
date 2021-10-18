

# PART-1 Train a Simple Neural Network using Microsoft Excel



# [MNIST Classification with less than 20,000 parameters and 99.4% validation accuracy](#part2)

***NETWORK***

![alt text](https://github.com/anirbanroy987/EVA7-TSAI-Learning-Modules-Phase-1/blob/main/images/NN.JPG?raw=true)


1. Drawing of the neural network diagram as shown in the figure.
2. Connecting all the neurons using arrows and mark them with appropriate names and values as shared in the session.
3. Explaining different mathematical equations using differential equations .
4. Run different learning rates and display the graphs -[0.1, 0.2, 0.5, 0.8, 1.0, 2.0] 


***FeedForward Equations - ***

**Input to Layer 1**

Hidden layers - 

h1=i1*w1 + i2*w2
h2=i1*w3 + i2*w4

We introduce a non-linear activation function i.e. to bring non-linearity into the model.
a_h1=σ(h1)=1/(1+exp(-h1))
a_h2=σ(h2)=1/(1+exp(-h2))

***Layer 1 to Layer 2***
_______

These include weights after the hidden layers .

o1=a_h1*w5+a_h2*w6
o2=a_h1*w7+a_h2*w8

a_o1=σ(o1)
a_o2=σ(o2)

***Layer 2 to Loss(MSE) ***
__________________

E1=1/2*(t1-a_o1)2
E2=1/2*(t2-a_o2)2
Etotal = E1+E2

***BACKPROPAGATION***

After initially getting our Errors we need to subtract and update our weights for each iteration so that we are able to predict and have a
good neural network .

In order to achieve this the below equations will serve as our code of operation using chain rule for 
all the weights .
As weights are responsible for predictions so we need to have a proper weights for a good predictive neural network.

1. Error --> Layer 1 

W5 - 

***∂E_t/∂w5 = ∂(E1+E2)/∂w5=∂E1/∂w5=(∂E1/a_o1) * (∂a_o1/∂o1)*(∂o1/∂w5)
∂E1/∂a_o1   = ∂(1/2*(t1-a_o1)^2)/∂a_o1=(t1-a_01)*(-1)=a_o1-t1
∂a_o1/do1   = ∂(σ(o1))/∂o1=σ(o1)*(1-σ(o1))=a_o1*(1-a_o1)
∂o1/∂w5     = a_h1
∂E_t/∂w5  = (a_o1-t1)*a_o1*(1-a_o1)*a_h1

W6 - 

***∂E_t/∂w6 = ∂(E1+E2)/∂w6=∂E1/∂w6=(∂E1/a_o1) * (∂a_o1/∂o1)*(∂o1/∂w6)
∂E1/∂a_o1   = ∂(1/2*(t1-a_o1)^2)/∂a_o1=(t1-a_01)*(-1)=a_o1-t1
∂a_o1/do1   = ∂(σ(o1))/∂o1=σ(o1)*(1-σ(o1))=a_o1*(1-a_o1)
∂o1/∂w6     = a_h2
∂E_t/∂w6  = (a_o1-t1)*a_o1*(1-a_o1)*a_h2

W7

***∂E_t/∂w7 = ∂(E1+E2)/∂w7=∂E1/∂w7=(∂E1/a_o2) * (∂a_o2/∂o1)*(∂o1/∂w7)
∂E1/∂a_o1   = ∂(1/2*(t2-a_o2)^2)/∂a_o2=(t2-a_o2)*(-1)=a_o2-t2
∂a_o1/do1   = ∂(σ(o1))/∂o1=σ(o1)*(1-σ(o1))=a_o2*(1-a_o2)
∂o1/∂w7     = a_h1
∂E_t/∂w7  = (a_o2-t2)*a_o2*(1-a_o2)*a_h1


W8

***∂E_t/∂w7 = ∂(E1+E2)/∂w8=∂E1/∂w5=(∂E1/a_o2) * (∂a_o2/∂o1)*(∂o1/∂w8)
∂E1/∂a_o1   = ∂(1/2*(t2-a_o2)^2)/∂a_o2=(t2-a_o2)*(-1)=a_o2-t2
∂a_o1/do1   = ∂(σ(o1))/∂o1=σ(o1)*(1-σ(o1))=a_o2*(1-a_o2)
∂o1/∂w8     = a_h2
∂E_t/∂w8  = (a_o2-t2)*a_o2*(1-a_o2)*a_h2


***Layer 2 to Layer 1***

W1

***∂E1/∂a_h1=(∂E1/a_o1) * (∂a_o1/∂o1)*(∂o1/∂a_h1)=(a_o1-t1)*a_o1*(1-a_o1)*w5
∂E2/∂a_h1=(∂E1/a_o1) * (∂a_o1/∂o1)*(∂o1/∂a_h1)=(a_o2-t2)*a_o2*(1-a_o2)*w7

***∂E1/∂a_h2=(a_o1-t1)*a_o1*(1-a_o1)*w6
∂E2/∂a_h2=(a_o2-t2)*a_o2*(1-a_o2)*w8

***∂E_T/∂a_h1=∂E1/∂a_h1+∂E2/∂a_h1

***∂E_t/dw1=∂E_t/∂a_h1*da_h1/∂h1 *∂h1/∂w1=[((a_o1-t1)*a_o1*(1-a_o1)*w5)+ ((a_o2-t2)*a_o2*(1-a_o2)*w7)*a_h1*(1-a_h1)*i1


W2

***∂E_t/dw2=∂E_t/∂a_h1*da_h1/∂h1 *∂h1/∂w2 =  [((a_o1-t1)*a_o1*(1-a_o1)*w5)+ ((a_o2-t2)*a_o2*(1-a_o2)*w7)*a_h1*(1-a_h1)*i2

W3

***∂E_t/dw3 = [((a_o1-t1)*a_o1*(1-a_o1)*w6)+ ((a_o2-t2)*a_o2*(1-a_o2)*w8)*a_h2*(1-a_h2)*i1

W4

***∂E_t/dw4 =  [((a_o1-t1)*a_o1*(1-a_o1)*w6)+ ((a_o2-t2)*a_o2*(1-a_o2)*w8)*a_h2*(1-a_h2)*i2  ***


We go the equations for the all the weights and now we just need to put in the values .


Starting with weight w5 (as it is connected to the last layer), calculate the equation of backpropagation. Trace the path from the output to the weight and use chain rule.

Change the final equation of w5 suitably to get w6, w7 and w8.
Again, trace the paths of the weights and put the appropriate variables in the equations.

Similarly, repeat the steps for w1, w2, w3 and w4. It will be easier to break down the chain rule into two steps, one from output to hidden layer and another from hidden to input layer.


Excel Screenshot of our values - 

![alt text](https://github.com/anirbanroy987/EVA7-TSAI-Learning-Modules-Phase-1/blob/main/images/NN_weight_update.JPG?raw=true)

Our Neural Network learns through a hyper parameter called learning rate - 
We have adjusted different learning rates and the below comaparison of learning rates is shown : -


![alt text](https://github.com/anirbanroy987/EVA7-TSAI-Learning-Modules-Phase-1/blob/main/images/NN_learning_rate.JPG?raw=true)


Conclusion : -

We can observe that higher the value of learning rate, higher the rate of convergence of loss for this particular problem. This is not true in most deep neural network problems as the learning rate is generally kept low to update the weights slowly.





# PART2 





