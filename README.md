# cnnGomoku
The program runs on cpu by default. To run on gpu, enable gpu mode of train() evaluate() in trainer.py and PolicyValueNet() in torchCNN.py\

More training games are in sgf.zip 

5/15
1.added rotated/flipped data. pytorch supports online data augmentation only for img but not tensor.
