import torch as t
import torchvision.datasets
from torch.utils.data import TensorDataset, DataLoader

class Data:
    def __init__(self):
        self.x_train: t.Tensor | None = None
        self.x_test: t.Tensor | None = None
        self.y_train: t.Tensor | None = None
        self.y_test: t.Tensor | None = None
        
    def load(self) -> None:
        mnist_train = torchvision.datasets.MNIST(
            root = "./",
            train = True,
            download = True
        )

        mnist_test = torchvision.datasets.MNIST(
            root = "./",
            train = False,
            download = True
        )

        self.x_train = mnist_train.data
        self.y_train = mnist_train.targets

        self.x_test = mnist_test.data
        self.y_test = mnist_test.targets

        self.x_train = self.x_train.float().reshape(-1, 28 * 28)
        self.x_test = self.x_test.float().reshape(-1, 28 * 28)

    def get_loaders(self, batch_size: int) -> tuple[DataLoader, DataLoader]:
        train_dataloader = DataLoader(
            dataset = TensorDataset(
                self.x_train,
                self.y_train
            ),
            batch_size = batch_size,
            shuffle = True
        )

        test_dataloader = DataLoader(
            dataset = TensorDataset(
                self.x_test,
                self.y_test
            ),
            batch_size = batch_size,
            shuffle = False
        )

        return train_dataloader, test_dataloader
