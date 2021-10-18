

# PART-1 Train a Simple Neural Network using Microsoft Excel



# [PART -2 MNIST Classification with less than 20,000 parameters and 99.4% validation accuracy](#part2)

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

This assignment aims to design a CNN model for MNIST Classification having 99.4% with following constraints.

- Total number of parameters should be less than 20000.
- The above accuracy should be achieved within 20 epochs.


The objective is to optimize a network combining many deep learning techniques on the layers and use very few parameters to achieve high accuracy.

***DATA PREPARATION***
We have created a dataloader variable and converted into to_tensor and normalized the values with mean and std dev - (0.1307,), (0.3081,).
Same approach was taken for test data .


### We have used the following approaches  - 

***NO DROPOUT ONLY BATCH NORM AND GAP WITHOUT DATA AUGMENTATION ***

1.Parameters - 9,186

2.Used GAP 

3.Used FC 

4.No dropout

5.Used Max Pooling 

6.Activation functions 

7.Learning rate - 0.01

8.Epochs - 20

9.BatchSize - 128


***ARCHITECTURE***

![alt text](https://github.com/anirbanroy987/EVA7-TSAI-Learning-Modules-Phase-1/blob/main/images/nodroput_onlygap.JPG?raw=true)


ACCURACY - 98.90

***USING DATA AUGMENTATION ,NO DROPOUT,BATCH SIZE***


```
train_data_with_augmentation = datasets.MNIST('./data', train=True, download=True,
                                transform=transforms.Compose([
                                    transforms.RandomRotation(5),
                                    transforms.RandomAffine(degrees=8, translate=(0.15,0.15), scale=(0.85, 1.15)),
                                    transforms.ColorJitter(brightness=0.1, contrast=0.1),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.1307,), (0.3081,))]))
```

1.Parameters - 9,186

2.Used GAP 

3.Used FC 

4.No dropout

5.Used Max Pooling 

6.Activation functions 

7.Learning rate - 0.01

8.Epochs - 20

9.BatchSize - 64

10.Data Augmentation

ACCURACY - 99.26%



### FINAL MODEL 

***With DROPOUT AND STEP WISE LR ***

Best Model using step wise learning rate scheduler and using a single dropout layer

1.Parameters - 9,320

2.Used GAP 

3.Used FC 

4.Dropout

5.Used Max Pooling 

6.Activation functions 

7.Learning rate - 0.01

8.Epochs - 18

9.BatchSize - 64

10.Data Augmentation

11.Stepwise LR scheduler -(step_size= 8, gamma= 0.25)

***ARCHITECTURE***

![alt text](https://github.com/anirbanroy987/EVA7-TSAI-Learning-Modules-Phase-1/blob/main/images/withdropout_nn.JPG?raw=true)







***MODEL LOGS : -***


Epoch: 1
loss=0.06085310876369476 batch_id=937: 100%|██████████| 938/938 [01:07<00:00, 13.90it/s]

TRAIN set: Average loss: 0.5584, Train Accuracy: 82.21%
TEST set: Average loss: 0.0766, Test Accuracy: 97.74%
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Epoch: 2
loss=0.06745391339063644 batch_id=937: 100%|██████████| 938/938 [01:07<00:00, 13.82it/s]

TRAIN set: Average loss: 0.1358, Train Accuracy: 96.14%
TEST set: Average loss: 0.0542, Test Accuracy: 98.32%
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Epoch: 3
loss=0.058390356600284576 batch_id=937: 100%|██████████| 938/938 [01:07<00:00, 13.94it/s]

TRAIN set: Average loss: 0.1063, Train Accuracy: 96.83%
TEST set: Average loss: 0.0627, Test Accuracy: 98.04%
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Epoch: 4
loss=0.020141618326306343 batch_id=937: 100%|██████████| 938/938 [01:07<00:00, 13.93it/s]

TRAIN set: Average loss: 0.0888, Train Accuracy: 97.39%
TEST set: Average loss: 0.0468, Test Accuracy: 98.50%
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Epoch: 5
loss=0.08200311660766602 batch_id=937: 100%|██████████| 938/938 [01:07<00:00, 13.91it/s]

TRAIN set: Average loss: 0.0816, Train Accuracy: 97.63%
TEST set: Average loss: 0.0348, Test Accuracy: 98.97%
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Epoch: 6
loss=0.09740690141916275 batch_id=937: 100%|██████████| 938/938 [01:06<00:00, 14.06it/s]

TRAIN set: Average loss: 0.0735, Train Accuracy: 97.83%
TEST set: Average loss: 0.0369, Test Accuracy: 98.89%
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Epoch: 7
loss=0.19831380248069763 batch_id=937: 100%|██████████| 938/938 [01:07<00:00, 13.98it/s]

TRAIN set: Average loss: 0.0708, Train Accuracy: 97.87%
TEST set: Average loss: 0.0292, Test Accuracy: 99.01%
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Epoch: 8
loss=0.010681346990168095 batch_id=937: 100%|██████████| 938/938 [01:07<00:00, 13.91it/s]

TRAIN set: Average loss: 0.0651, Train Accuracy: 98.03%
TEST set: Average loss: 0.0328, Test Accuracy: 98.99%
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Epoch: 9
loss=0.06247676908969879 batch_id=937: 100%|██████████| 938/938 [01:07<00:00, 13.99it/s]

TRAIN set: Average loss: 0.0523, Train Accuracy: 98.42%
TEST set: Average loss: 0.0222, Test Accuracy: 99.36%
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Epoch: 10
loss=0.0034355116076767445 batch_id=937: 100%|██████████| 938/938 [01:06<00:00, 14.01it/s]

TRAIN set: Average loss: 0.0503, Train Accuracy: 98.56%
TEST set: Average loss: 0.0209, Test Accuracy: 99.33%
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Epoch: 11
loss=0.00834533479064703 batch_id=937: 100%|██████████| 938/938 [01:06<00:00, 14.04it/s]

TRAIN set: Average loss: 0.0483, Train Accuracy: 98.55%
TEST set: Average loss: 0.0205, Test Accuracy: 99.38%
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Epoch: 12
loss=0.05023888498544693 batch_id=937: 100%|██████████| 938/938 [01:07<00:00, 13.79it/s]

TRAIN set: Average loss: 0.0492, Train Accuracy: 98.53%
TEST set: Average loss: 0.0204, Test Accuracy: 99.35%
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Epoch: 13
loss=0.01739363744854927 batch_id=937: 100%|██████████| 938/938 [01:07<00:00, 13.87it/s]

TRAIN set: Average loss: 0.0474, Train Accuracy: 98.56%
TEST set: Average loss: 0.0199, Test Accuracy: 99.35%
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Epoch: 14
loss=0.011957034468650818 batch_id=937: 100%|██████████| 938/938 [01:06<00:00, 14.02it/s]

TRAIN set: Average loss: 0.0453, Train Accuracy: 98.62%
TEST set: Average loss: 0.0227, Test Accuracy: 99.32%
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Epoch: 15
loss=0.10129786282777786 batch_id=937: 100%|██████████| 938/938 [01:06<00:00, 14.06it/s]

TRAIN set: Average loss: 0.0474, Train Accuracy: 98.54%
TEST set: Average loss: 0.0210, Test Accuracy: 99.31%
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Epoch: 16
loss=0.08632291853427887 batch_id=937: 100%|██████████| 938/938 [01:07<00:00, 13.92it/s]

TRAIN set: Average loss: 0.0461, Train Accuracy: 98.58%
TEST set: Average loss: 0.0218, Test Accuracy: 99.28%
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Epoch: 17
loss=0.006729969754815102 batch_id=937: 100%|██████████| 938/938 [01:07<00:00, 13.86it/s]

TRAIN set: Average loss: 0.0420, Train Accuracy: 98.72%
TEST set: Average loss: 0.0206, Test Accuracy: 99.37%
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Epoch: 18
loss=0.04360434412956238 batch_id=937: 100%|██████████| 938/938 [01:07<00:00, 13.92it/s]

TRAIN set: Average loss: 0.0414, Train Accuracy: 98.77%
TEST set: Average loss: 0.0201, Test Accuracy: 99.38%
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Epoch: 19
loss=0.11143185943365097 batch_id=937: 100%|██████████| 938/938 [01:07<00:00, 14.00it/s]

TRAIN set: Average loss: 0.0437, Train Accuracy: 98.67%
***TEST set: Average loss: 0.0197, Test Accuracy: 99.43%***
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~





