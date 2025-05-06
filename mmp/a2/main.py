from typing import Tuple
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader

# these are the labels from the Cifar10 dataset:
CLASSES = (
    "plane",
    "car",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)


class MmpNet(nn.Module):
    """Exercise 2.1"""
    def __init__(self, num_classes: int):
        super().__init__()
        self.model = torchvision.models.mobilenet_v2(weights = "DEFAULT")
        self.model.classifier[1].out_features = num_classes

    def forward(self, x: torch.Tensor):
        out = self.model(x)
        return out


def get_dataloader(
    is_train: bool, data_root: str, batch_size: int, num_workers: int
) -> DataLoader:
    """Exercise 2.2

    @param is_train: Whether this is the training or validation split
    @param data_root: Where to download the dataset to
    @param batch_size: Batch size for the data loader
    @param num_workers: Number of workers for the data loader
    """

    dataset = torchvision.datasets.CIFAR10(
        root = data_root,
        train = is_train,
        download = True,
        transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize((256, 256)),
            torchvision.transforms.CenterCrop(224),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    )

    data_loaders = DataLoader(
        dataset, batch_size=batch_size,
        shuffle=is_train,
        drop_last=is_train,
        num_workers=num_workers)
    return data_loaders

def get_criterion_optimizer(model: nn.Module) -> Tuple[nn.Module, optim.Optimizer]:
    """Exercise 2.3a

    @param model: The model that is being trained.
    @return: Returns a tuple of the criterion and the optimizer.
    """
    lern_rate = 0.004
    error_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lern_rate)
    return (error_func, optimizer)


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
):
    """Exercise 2.3b

    @param model: The model that should be trained
    @param loader: The DataLoader that contains the training data
    @param criterion: The criterion that is used to calculate the loss for backpropagation
    @param optimizer: Executes the update step
    @param device: The device where the epoch should run on
    """

    model = model.to(device)
    model.train()

    for i, data in enumerate(loader):
        print(f"Training batch {i}")
        images, labels = data

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        prediction = model(images)
        loss = criterion(prediction, labels)
        loss.backward()
        optimizer.step()

def eval_epoch(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    """Exercise 2.3c

    @param model: The model that should be evaluated
    @param loader: The DataLoader that contains the evaluation data
    @param device: The device where the epoch should run on

    @return: Returns the accuracy over the full validation dataset as a float."""

    model = model.to(device)
    model.eval()
    total = 0
    correct = 0
    total_loss = 0

    criterion = get_criterion_optimizer(model)[0]

    with torch.no_grad():
        for i, data in enumerate(loader):
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)

            prediction = model(images)
            loss = criterion(prediction, labels)
            total += images.shape[0]
            total_loss += loss.item() * images.shape[0]
            predicted_label = prediction.argmax(dim=-1)
            correct += torch.sum(predicted_label == labels)

    loss = total_loss / total
    accuracy = correct / total

    print(f"Loss: {loss:.4f}, Accuracy: {accuracy * 100:.4f}")


def main():
    """Exercise 2.3d"""

    model = MmpNet(len(CLASSES))
    training_dataloader = get_dataloader(True, "data", 64, 1)
    loss_func, optimizer = get_criterion_optimizer(model)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    for i in range(1):
        print(f"Training epoc {i}")
        train_epoch(model, training_dataloader, loss_func, optimizer, device)
    
    testing_dataloader = get_dataloader(False, "data", 1, 1)
    
    eval_epoch(model, testing_dataloader, device)


if __name__ == "__main__":
    main()
