# Leren & Beslissen 2020

## *Exploring the border between deep and shallow learning*

### Summary:

I am supervising the Leren & Beslsissen course of 2020. This repository functions as a provided-code hub for the students.
It contains example models, dataloaders and training scripts in PyTorch, as well as, meeting notes and assignment instructions.


### Classes:

##### model-classes
Currently there are two convnets for image-data (lenet5 and resnet), two recurrent-nets for text-data (rnn and lstm) and one fully connected (fc) for any data-type. 

##### data-loaders
Currently contains MNIST, stanford-sentiment and an example toy dataset to show how you can make your own.

##### NNClassificationTrainer
Is a class with a pre-made training script as an example for classification.
If regression or something else is endeavoured, then it will need to be altered by student, it functions more as an example. 
Can be used like the shallow classes in sklearn where you just call *model.fit(x, y)*. However, here it takes in as arguments at initialisation: 
- the train-dataloader
- the test-dataloader
- optimizer
- model 
- number of epochs
- loss-function
- device. 

Thereafter you call *.train()* instead.

### Build and run:

- Ensure you have pip3 installed

> sudo apt install python3-pip

- Install virtualenv

> sudo pip3 install virtualenv 

- Create env

> virtualenv -p python3 ~/virtualenvs/NAME --system-site-packages

- Activate

> source ~/virtualenvs/NAME/bin/activate 

- Install requirements

> pip3 install -r requirements.txt 

- If you use an IDE, in the settings set project-interpreter to:

> ~/virtualenvs/NAME/bin/python3.7

- Run

> python3 example.py

### See also:

- Example on [colab](https://colab.research.google.com/drive/1arq7ZpWoO4Xw1od_RbMTHCl5IwIoYXOZ) 

### Updates:

...