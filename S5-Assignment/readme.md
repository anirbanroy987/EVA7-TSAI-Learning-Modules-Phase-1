***new target is:***

1.99.4% (this must be consistently shown in your last few epochs, and not a one-time achievement)

2.Less than or equal to 15 Epochs

3.Less than 10000 Parameters (additional points for doing this in less than 8000 pts)


# STEP - 1

***Target***

To create a basic skeleton of the the model

Parameters: Less than 10,000

Data Augmentations: None

Regularization: None

LR Scheduler: None

No. of Epochs: less than 20

***Model Results***

Total Parameters used: 9,902

Best Train Accuracy: 99.28%

Best Test Accuracy: 99.12%

Consistency: Did not achieve any consistency with test accuracy greater than 99.4. How ever tried to run till 20 epochs to understand if our model performs well but it deosnt go beyond 99.1 %.

***Analysis***

With the base network architecture, we achieved a test accuracy of 99% which is not even close to the target test accuracy to achieve. We plan to improve the test accuracy by doing the following approaches:

Split the process into two different networks by varying the number of parameters.

Keep the same network architecture, but removing the fully connected layers. Reduce the number of parameters in the model.



# STEP-2

***Target***



Parameters: Less than 10,000

Data Augmentations: No

Regularization: Yes

LR Scheduler: No

No. of Epochs: 14

***Model Results***

Total Parameters used: 9,060

Best Train Accuracy: 99.05%

Best Test Accuracy: 99.10%

***Analysis ***

Consistency: Did not achieve any consistency with test accuracy greater than 99.10
Added Dropout Layer to the network and remove FC layers .

we can see some underfitting in the model in the next model lets try to use Data Augmentation and Step wise LR .


# STEP -3

***Target***

Parameters: Less than 10,000

Data Augmentations: Yes

Regularization: Yes

LR Scheduler: Yes

No. of Epochs: 20



***Model Results***

Total Parameters used: 9,060


Best Train Accuracy: 99.06%

Best Test Accuracy: 99.37%

Consistency: Accuracy greater than 99.4 consistent from 9th epoch to 14th epoch

Data Augmentation: Random Rotation of 7 degrees.

Learning Rate Scheduler: StepLR used

***Analysis***

The StepLR helped in stabilizing the learning, by reducing the learning rate to 10% after every 6th Epochs.

Why after every 6 epochs? We observed that the Loss was bouncing up and down, from 6th/7th epoch, so reducing the LR at that point would made the training stable.

The Random Rotation was applied with ±7 degrees.



# STEP - 4 

***Targets***


To create a model with higher Recpetive Field than 28

Parameters: Less than 8,000

Data Augmentations: RandomRotation ±15°

Regularization: DropOut

LR Scheduler: StepLR

No. of Epochs: 20


***Model Results***

Total Parameters Used: 7,946

Train Accuracy: 99.36

Test Accuracy: 99.52

Consistent From: 10th Epoch to End

Data Augmentation: Randam Rotation of ±15°

LR Scheduler: StepLR

***Analysis***

The Increase in Receptive Field may have helped, as we were able to cross the 99.5% barrier and the model is even more stable now.
