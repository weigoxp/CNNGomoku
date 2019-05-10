import sys
import torch.utils.data as data
from SGFFileProcess import *
import torch.nn.functional as F
from draw import *
import torch
import numpy as np
from datapacker import dataloader



cuda = torch.device('cuda')
from torchCNN import *
sgf = SGFflie()
nnn = PolicyValueNet()

def train():
    loader = dataloader('training/',4)
    for step, (board, nextmove) in enumerate(loader):

        # feed in nn
        nnn.train(board.float().cuda(),nextmove.cuda())
        # only used to print label distribution. only an approximate since b/w has different value.
        # tmp = np.array(move[k]).reshape(15, 15)
        # labelmatrix += tmp
        print(" Training: ", step, end='\r')

    return

def evaluate(path):
    correct =0
    total = 0

    loader = dataloader(path, 1)
    for step, (board, nextmove) in enumerate(loader):

        predict = nnn.classify(board.float().cuda())
        # print(predict)
        # print(nextmove.item())

        correct += 1 if (predict == nextmove.item()) else 0
        total += 1
        print("Evaluating: ", step, end='\r')

    return  (correct/total)


if __name__ == '__main__':

    #np.set_printoptions(suppress=True)
    # labelmatrix = [[0]*15 for i in range(15)]
    # predictmatrix = [[0]*15 for i in range(15)]

    accuracies = [[],[]]

    for i in range(100):
        ## begin to train data.
        train()
        print("\nEpoch done", i)

        #  nnn.policy_value_net = torch.load('model.pth')
        # evaluate every 2 epochs

        if i%2 ==0:
            training_accuracy = evaluate('training/')
            testing_accuracy = evaluate('testing/')

            print("\ntraining_accuracy" ,training_accuracy)
            print("\ntesting_accuracy",testing_accuracy)

            #draw
            accuracies[0].append(training_accuracy)
            accuracies[1].append(testing_accuracy)

            draw(accuracies)

    torch.save(nnn.policy_value_net, 'model.pth')

  #  print(labelmatrix)

    # np.savetxt('labels.txt', labelmatrix,'%5.0f', delimiter=',')
    # np.savetxt('predict.txt', predictmatrix, delimiter=',')






