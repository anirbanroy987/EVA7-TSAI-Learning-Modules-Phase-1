
Write a neural network that can:
take 2 inputs:
an image from the MNIST dataset (say 5), and
a random number between 0 and 9, (say 7)
and gives two outputs:
the "number" that was represented by the MNIST image (predict 5), and
the "sum" of this number with the random number and the input image to the network (predict 5 + 7 = 12)



SOLUTION : -


1.Data Representation


A Custom Dataset CLASS RandomNumGen is created which take Inputs(train or test). The MNIST Data is given as the input .
WE have also created a __getitem__() method, which will index the MNIST data and give us a image and its corresponding label
and also using the randint function we will get a random number.The Random Number is than converted to a one hot encoded vector


We have generated two tuples from the class we have created: -

 a. the random number + MNIST image is sent as a tuple of input from the class.


 b. MNIST label and the sum of MNIST label the random number genrated is added and the result along with MNIST label .


train_ds and test dataset will be created using the RandomNumGen class .


Screenshot of the code: -


##class RandomNumGen(Dataset):
  
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


Neural Architecture
________________
![Alt text](https://github.com/anirbanroy987/EVA7-TSAI-Learning-Modules-Phase-1/blob/main/images/Neural_architecture.JPG?raw=true "Optional Title")
