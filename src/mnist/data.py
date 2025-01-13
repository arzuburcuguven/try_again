from pathlib import Path
import torch
import typer
from torch.utils.data import Dataset
from . import _PATH_DATA_RAW, _PATH_DATA_PROCESSED


def normalizer(images: torch.Tensor) -> torch.Tensor:
    """Normalize images."""
    return (images - images.mean()) / images.std()


class MyDataset(Dataset):
    """My custom dataset."""

    def __init__(self, raw_data_path: Path = _PATH_DATA_RAW, processed_dir: Path = _PATH_DATA_PROCESSED) -> None:
        self.data_path = raw_data_path
        self.processed_dir = processed_dir
        # load data from raw_data_path
        train_images, train_target = [], []
        # how to use this raw_data_path to properly get data?
        for i in range(6):
            train_images.append(torch.load(f"{raw_data_path}/train_images_{i}.pt"))
            train_target.append(torch.load(f"{raw_data_path}/train_target_{i}.pt"))
        # for test it is simpler
        test_images = torch.load(f"{raw_data_path}/test_images.pt")
        test_target = torch.load(f"{raw_data_path}/test_target.pt")
        # concat
        train_images = torch.cat(train_images)
        train_target = torch.cat(train_target)

        self.train_images = train_images.unsqueeze(1).float()
        self.test_images = test_images.unsqueeze(1).float()
        self.train_target = train_target.long()
        self.test_target = test_target.long()

    def __len__(self) -> int:
        """Return the length of the dataset."""
        return self.train_images.shape[0]

    def __getitem__(self, index: int):
        """Return a given sample from the dataset."""
        return self.train_images[index], self.train_target[index]

    def preprocess(self) -> None:
        """Preprocess the raw data and save it to the output folder."""
        train_images = normalizer(self.train_images)
        test_images = normalizer(self.test_images)

        torch.save(self.train_images, f"{self.processed_dir}/train_images.pt")
        torch.save(self.train_target, f"{self.processed_dir}/train_target.pt")
        torch.save(self.test_images, f"{self.processed_dir}/test_images.pt")
        torch.save(self.test_target, f"{self.processed_dir}/test_target.pt")

        train_dataset = torch.utils.data.TensorDataset(train_images, self.train_target)
        test_dataset = torch.utils.data.TensorDataset(test_images, self.test_target)

        return train_dataset, test_dataset

def preprocess(raw_data_path: Path = _PATH_DATA_RAW, processed_dir: Path = _PATH_DATA_PROCESSED) -> None:
    print("Preprocessing data...")
    dataset = MyDataset(raw_data_path, processed_dir)
    dataset.preprocess()


if __name__ == "__main__":
    typer.run(preprocess)
