from mnist.train import train
import os


def test_my_training():
    train(1e-3, 32, 10)
    assert os.path.exists("models/model.pth")
