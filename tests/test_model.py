import torch
from src.mnist.model import MyAwesomeModel


def test_my_model():
    input = torch.randn(1, 1, 28, 28)
    model = MyAwesomeModel()
    output = model(input)
    assert output.shape == (1, 10)
