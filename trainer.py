import sys

from SGFFileProcess import *
import torch.nn.functional as F

import torch
import numpy as np
cuda = torch.device('cuda')
from torchCNN import *
sgf = SGFflie()
nnn = PolicyValueNet()

def train(path):

    step =0
    for filepath in path:
        board, nextmove = sgf.createTraindataFromqipu(filepath)
        for k in range(len(board)):
            # 1d array to 4d tensor.
            x = np.array(board[k]).reshape(1,1, 15, 15)
            x = torch.from_numpy(x)

            y = np.array([np.argmax(nextmove[k])])
            y = torch.from_numpy(y)
            # feed in nn
            nnn.train(x.cuda().float(),y.cuda())

            # only used to print label distribution. not very accurate since b/w has different value.
            # tmp = np.array(move[k]).reshape(15, 15)
            # labelmatrix += tmp

        print("\r Training: ", step, end='')
        step += 1
    return

def evaluate(path):
    correct =0
    total = 0
    step = 0

    for filepath in path:
        board, nextmove = sgf.createTraindataFromqipu(filepath)
        for k in range(len(board)):
            x = np.array(board[k]).reshape(1, 1, 15, 15)
            x = torch.from_numpy(x)

            y = np.array([np.argmax(nextmove[k])])
            y = torch.from_numpy(y)

            predict = nnn.classify(x.float().cuda())

            tmp = np.array(nextmove[k]).reshape(15, 15)
            # predictmatrix += predict

            correct += 1 if (predict == y.item()) else 0
            total += 1
            print("\r Evaluating: ", step, end='')

        step += 1
    return  (correct/total)




if __name__ == '__main__':
    np.set_printoptions(suppress=True)
    labelmatrix = [[0]*15 for i in range(15)]
    predictmatrix = [[0]*15 for i in range(15)]
    testing = sgf.allFileFromDir('testing/')
    training = sgf.allFileFromDir('training/')
    ## begin to train data.

    for i in range(1):
        train(training)
        print("Epoch done")
    torch.save(nnn.policy_value_net, 'model.pth')

  #  print(labelmatrix)

    # np.savetxt('labels.txt', labelmatrix,'%5.0f', delimiter=',')


   # evaluate training data.
    #  nnn.policy_value_net = torch.load('model.pth')

    training_accuracy = evaluate(training)
    testing_accuracy = evaluate(testing)

    print(training_accuracy)
    print(testing_accuracy)

    # np.savetxt('predict.txt', predictmatrix, delimiter=',')