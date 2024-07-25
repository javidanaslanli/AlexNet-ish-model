# AlexNetish model with Fully Sharded Data Parallel (FSDP)
This repository contains an implementation of a custom AlexNet-ish model using PyTorch with Fully Sharded Data Parallel (FSDP) for efficient training on large datasets. The model can be trained and validated on a custom dataset.

## About my results and my training

The architecture of the model is nearly same as the original Alexnet (https://proceedings.neurips.cc/paper_files/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf ---> you can read from here) but the dataset I used for training (https://www.kaggle.com/datasets/gpiosenka/100-bird-species) started to overfit with the same parameters as they used in original model.
So I decided to make some changes, that changes made model to reach 62% accuracy on the data that after 30 epochs. It is a bird species classification dataset that contains 525 different classes of birds. I tried different hyperparameters and 64 batch size and 0.001 learning rate worked best for me. 25 minibatches of batches to train took nearly 412 ms, also, full epoch took nearly 300 second to train. I trained model on kaggle T4's with FSDP and they used 5.8 gb per GPU. 

## Changes I made

Firstly, original paper they use Local response normalization and I sticked to LRN in my model, but I changed the number of linear layers which is 3 in original paper but I used only 2. Because I found that 3 linear layers were causing overfitting in my model. Second change I made is to use AdamW optimizer rather than SGD for speed. I also, changed the convolutional layers parameters to decrease the number of parameters in model (it was too high and causing overfitting). I did not use PCA data preprocessing as the used in the paper but I added some noise o data with using random transformations from torchvision. I used 227x227 as input but first I resized images to 256x256 then I get an center crop from image 227x227.

![Screenshot 2024-07-25 162636](https://github.com/user-attachments/assets/65516193-f975-43b9-8b91-af2fd82c5f21)

## Requirements

To install the necessary dependencies, run:

```sh
pip install -r requirements.txt
```



