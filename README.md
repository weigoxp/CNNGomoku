# cnnGomoku

*Gomoku game training model based on historic pro games and Covolutional neural network*

You need pytorch to run this program: https://pytorch.org/

The program runs on cpu by default. To run on CUDA, enable gpu mode of train() evaluate() in trainer.py and PolicyValueNet() in torchCNN.py\

More training games are in sgf.zip 

Updates:

5/15/2019
1.  added rotated/flipped data. pytorch supports online data augmentation only for img but not tensor.
