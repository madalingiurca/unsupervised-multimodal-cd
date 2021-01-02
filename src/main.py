from src.dataPrepare import process_image
from matplotlib import pyplot as plt
from neuralNet.neuralNetwork import XNet
import torch

if __name__ == '__main__':
    xnet = XNet()

    roi, t1, t2 = process_image()
    sample = torch.tensor(t1[0][0]).permute(2, 0, 1).unsqueeze(0)
    print(sample.shape)
    sample = xnet.CNNLayerY(sample)
    print(sample.shape)
