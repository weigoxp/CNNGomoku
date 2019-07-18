# cnnGomoku

*Gomoku game training model based on historic pro games and Covolutional neural network*

See cnndes.pdf for detailed description.

1. You need pytorch to run this program: https://pytorch.org/

2. The program runs on cpu by default. To run on CUDA, enable gpu mode of train() evaluate() in trainer.py and PolicyValueNet() in torchCNN.py\

3. More training games are in sgf.zip 

Updates:

5/15/2019
1.  added rotated/flipped data. pytorch supports online data augmentation only for img but not tensor.
