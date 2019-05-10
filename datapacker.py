import torch.utils.data as data
import torch
import numpy as np
from SGFFileProcess import *


sgf = SGFflie()


def dataloader(path, MINIBATCH_SIZE):

    board, nextmove = sgf.createTraindata(path)
    print("data loaded, board nums: ", len(board))

    # transform data from here.
    x = np.array(board).reshape(len(board), 1, 15, 15)
    x = torch.from_numpy(x)
    for i in range(len(nextmove)):
        nextmove[i] = np.argmax(nextmove[i])
    y = torch.from_numpy(np.array(nextmove))

    torch_dataset = data.TensorDataset(x, y)

    # put the dataset into DataLoader
    loader = data.DataLoader(
        dataset=torch_dataset,
        batch_size=MINIBATCH_SIZE,
        shuffle=True,
        num_workers=2,
        drop_last=False
    )
    return loader
