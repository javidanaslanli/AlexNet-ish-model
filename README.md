# AlexNetish model with Fully Sharded Data Parallel (FSDP)
This repository contains an implementation of a custom AlexNet-ish model using PyTorch with Fully Sharded Data Parallel (FSDP) for efficient training on large datasets. The model can be trained and validated on a custom dataset.

## About my results and my training

The architecture of the model is nearly same as the original Alexnet (https://proceedings.neurips.cc/paper_files/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf ---> you can read from here) but the dataset I used for training (https://www.kaggle.com/datasets/gpiosenka/100-bird-species) started to overfit with the same parameters as they used in original model.
So I decided to make some changes, that changes made model to reach 62% accuracy on the data that after 30 epochs. It is a bird species classification dataset that contains 525 different classes of birds. I found the best 
