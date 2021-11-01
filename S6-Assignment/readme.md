
# 1. What is your code all about, (what the notebook does; what model.py is doing).

![alt text](https://github.com/anirbanroy987/EVA7-TSAI-Learning-Modules-Phase-1/blob/main/images/code%20contains.JPG?raw=true)

We have modularized our code based on different types of batch normalization techniques to be used .
And each time we run our model we need to just put in our parameter as whether it should run as group ,batch norm or as layern norm.


# 2 .Write a single notebook file to run all the 3 models above for 20 epochs each.

The file Assignment_6_normalization.ipynb contains all the three models based on the parameterization given as above for three regularization input.

# 3.your findings for normalization techniques.

  a.We found that GroupNorm was better than batch norm when we have smaller batch size.
  b.The accuracy in Group Norm was better .
  c.Layernorm performance was lower compared to other two techniques,we think that with regularization the model might improve.
  
 # 4.Loss and accuracy graphs of three techniques used.
***LOSS***

 ![alt text](https://github.com/anirbanroy987/EVA7-TSAI-Learning-Modules-Phase-1/blob/main/images/loss_metrics.JPG?raw=true)
 
 ***ACCURACY***
 
 ![alt text](https://github.com/anirbanroy987/EVA7-TSAI-Learning-Modules-Phase-1/blob/main/images/accuracy_metrics.JPG?raw=true)
 
 # Misclassified Predictions for all the three Techniques.
 Batch Norm 
  ![alt text](https://github.com/anirbanroy987/EVA7-TSAI-Learning-Modules-Phase-1/blob/main/images/batch_norm%2Bl1.JPG?raw=true)
  
  Group Norm 
  
   ![alt text](https://github.com/anirbanroy987/EVA7-TSAI-Learning-Modules-Phase-1/blob/main/images/group_norm.JPG?raw=true)
   
   Layer Norm
   
    ![alt text](https://github.com/anirbanroy987/EVA7-TSAI-Learning-Modules-Phase-1/blob/main/images/layer_norm_predictions.JPG?raw=true)
