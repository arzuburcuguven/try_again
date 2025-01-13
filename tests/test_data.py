from torch.utils.data import Dataset

from src.mnist.data import MyDataset
from tests import _PATH_DATA_RAW, _PATH_DATA_PROCESSED
import torch


def test_my_dataset():
    """Test the MyDataset class."""
    dataset = MyDataset(_PATH_DATA_RAW, _PATH_DATA_PROCESSED)
    assert isinstance(dataset, Dataset)


def test_data():
    dataset = MyDataset(_PATH_DATA_RAW, _PATH_DATA_PROCESSED)
    train, test = dataset.preprocess()
    assert len(train) == 30000
    assert len(test) == 5000
    for dataset in [train, test]:
        for x, y in dataset:
            assert x.shape == (1, 28, 28)
            assert y in range(10)
    train_targets = torch.unique(train.tensors[1])
    assert (train_targets == torch.arange(0, 10)).all()
    test_targets = torch.unique(test.tensors[1])
    assert (test_targets == torch.arange(0, 10)).all()
