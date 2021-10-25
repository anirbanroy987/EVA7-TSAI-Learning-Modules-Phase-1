***Your new target is:
1.99.4% (this must be consistently shown in your last few epochs, and not a one-time achievement)
2.Less than or equal to 15 Epochs
3.Less than 10000 Parameters (additional points for doing this in less than 8000 pts)***


***STEP - 1***

***Target

To create a basic skeleton of the the model

Parameters: Less than 10,000

Data Augmentations: None

Regularization: None

LR Scheduler: None

No. of Epochs: less than 20

***Model Results : - 

Total Parameters used: 9,902

Best Train Accuracy: 99.28%

Best Test Accuracy: 99.12%

Consistency: Did not achieve any consistency with test accuracy greater than 99.4. How ever tried to run till 20 epochs to understand if our model performs well but it deosnt go beyond 99.1 %.

***Analysis:

With the base network architecture, we achieved a test accuracy of 99% which is not even close to the target test accuracy to achieve. We plan to improve the test accuracy by doing the following approaches:

Split the process into two different networks by varying the number of parameters.

Keep the same network architecture, but removing the fully connected layers. Reduce the number of parameters in the model.
