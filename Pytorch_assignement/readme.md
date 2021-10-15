
Write a neural network that can take 2 inputs:


a. an image from the MNIST dataset (say 5),

b. a random number between 0 and 9, (say 7)

and gives two outputs:

1.the "number" that was represented by the MNIST image (predict 5), and
2.the "sum" of this number with the random number and the input image to the network (predict 5 + 7 = 12)



**SOLUTION : -**


__1.Data Representation__


A Custom Dataset CLASS RandomNumGen is created which take Inputs(train or test). The MNIST Data is given as the input .
WE have also created a __getitem__() method, which will index the MNIST data and give us a image and its corresponding label
and also using the randint function we will get a random number.The Random Number is than converted to a one hot encoded vector


We have generated two tuples from the class we have created: -

 a. the random number + MNIST image is sent as a tuple of input from the class.


 b. MNIST label and the sum of MNIST label the random number genrated is added and the result along with MNIST label .


train_ds and test dataset will be created using the RandomNumGen class .


Screenshot of the code: -


class RandomNumGen(Dataset):
  
    def __init__(self, MNIST_data):
        self.MNIST = MNIST_data
       
    
    def __getitem__(self, index):
        image, label = self.MNIST[index]
        number = random.randint(0,9)
        
        onehot_num = torch.zeros(10)
        onehot_num [number] = 1
       

        return ((image,onehot_num ), (label,number+label))

    def __len__(self):
        return len(self.MNIST)


We have done this as this becomes a classification problem and we can use a suitable classification loss function like cross entrophy or negative lok-likelihood functions.


***Neural Architecture***
________________
![Alt text](https://github.com/anirbanroy987/EVA7-TSAI-Learning-Modules-Phase-1/blob/main/images/Neural_architecture.JPG?raw=true "Optional Title")






The network has 7 convoluted layers and 2 max pooling after every 2 conv layers and bringing the size to 28x28 to 1x1.

 The model layers are: 
Network(
  (conv1): Conv2d(1, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  
  (conv2): Conv2d(16, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  
  (pool1): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  
  (conv3): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  
  (conv4): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  
  (pool2): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  
  (conv5): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1))
  
  (conv6): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1))
  
  (conv7): Conv2d(256, 10, kernel_size=(3, 3), stride=(1, 1))
  
  (layer1): Linear(in_features=20, out_features=19, bias=False)
  
)


***Data Formation from two inputs ***
_________________________

    def forward(self, x, y):
        x = self.pool1(F.relu(self.conv2(F.relu(self.conv1(x)))))
        x = self.pool2(F.relu(self.conv4(F.relu(self.conv3(x)))))
        x = F.relu(self.conv6(F.relu(self.conv5(x))))
        x = self.conv7(x)
        x = x.view(-1, 10)

        y = torch.cat((x,y),dim=1) #Horizontal Stacking. 1x10 and 1x10 -> 1x20


We have concatenated the final probabilities of 10 classes for a MNIST image from final convulated layer 
 with the Random number represented in 10-bit one hot encoded vector.
After this we get a tensor of size 20, which we pass through final fully connected layer (self.layer1) 
with input_features = 20 and output as 19(to get possible probabilities of sums which are 19 ).

 output_features = 19, to get all possible probabilities of sums which are 19 (0 → 18).


**GPU Usage**

```
def get_default_device():
    """
    Pick GPU if available, else CPU.
    """
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')

device = get_default_device()
```


***Loss Functions Reasons***

1.This is a multiclass classification problem , we have used NLL with a combination of log_softmax ..
Why log_softmax as basically, nn.NLLLoss expects log probabilities as input instead of probabilities.
Equation .
![Alt text](https://github.com/anirbanroy987/EVA7-TSAI-Learning-Modules-Phase-1/blob/main/images/Loss_function.JPG?raw=true "Loss Function Equations")

when training a model, we aspire to find the minima of a loss function given a set of parameters (in a neural network, these are the weights and biases). We can interpret the loss as the “unhappiness” of the network with respect to its parameters. The higher the loss, the higher the unhappiness: we don’t want that. We want to make our models happy.

So if we are using the negative log-likelihood as our loss function, when does it become unhappy? And when does it become happy?
The negative log-likelihood becomes unhappy at smaller values, where it can reach infinite unhappiness (that’s too sad), and becomes less unhappy at larger values. Because we are summing the loss function to all the correct classes, what’s actually happening is that whenever the network assigns high confidence at the correct class, the unhappiness is low, but when the network assigns low confidence at the correct class, the unhappiness is high.

![alt text](https://github.com/anirbanroy987/EVA7-TSAI-Learning-Modules-Phase-1/blob/main/images/NLL_image.JPG?raw=true)




