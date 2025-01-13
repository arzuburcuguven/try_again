import torch
from src.mnist.train import train
from src.mnist.data import MyDataset
import os

def test_my_training():
    train(1e-3, 32, 10)
    assert os.path.exists(os.path.abspath("/Users/argy/workspace/try_again/models/model.pth"))

