import torch
import torch.optim as optim
import torchvision
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
from .model import MmpNet
from ..a4 import dataset
from ..a4 import anchor_grid
from . import model as md
from ..a2 import main as a2

curr_epoch = 0
curr_batch = 0

def step(
    model: MmpNet,
    criterion,
    optimizer: optim.Optimizer,
    img_batch: torch.Tensor,
    lbl_batch: torch.Tensor,
) -> float:
    """Performs one update step for the model

    @return: The loss for the specified batch. Return a float and not a PyTorch tensor
    """
    raise NotImplementedError()

def get_tensorboard_writer(model_name):
    current_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    tensorboard_writer = SummaryWriter(log_dir=f"tensorboard_logs/{model_name}_{current_time}")
    return tensorboard_writer


def get_random_sampling_mask(labels: torch.Tensor, neg_ratio: float) -> torch.Tensor:
    """
    @param labels: The label tensor that is returned by your data loader.
    The values are either 0 (negative label) or 1 (positive label).
    @param neg_ratio: The desired negative/positive ratio.
    Hint: after computing the mask, check if the neg_ratio is fulfilled.
    @return: A tensor with the same shape as labels
    """
    assert labels.min() >= 0 and labels.max() <= 1  # remove this line if you want
    raise NotImplementedError()

def train_epoch(model, loader: dataset.DataLoader, criterion, optimizer, device):
    
    global curr_batch
    global curr_epoch

    model = model.to(device)
    model.train()

    for batch in loader:

        images, labels, ids = batch

        images = images.to(device)
        labels = labels.to(device)

        loss = md.step(model, criterion, optimizer, images, labels.float())
        print(f"e:{curr_epoch}/b:{curr_batch}/l:{loss}")
        curr_batch += 1

def val_epoch(val_epoch, model, loader: dataset.DataLoader, device: torch.device, tensorboard_writer) -> float:

    model = model.to(device)
    model.eval()
    total = 0
    correct = 0
    total_loss = 0

    criterion = a2.get_criterion_optimizer(model)[0]

    with torch.no_grad():
        for i, data in enumerate(loader):
            images, labels, ids = data
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

    print(f"Validation epoch {val_epoch}, Loss: {loss:.4f}, Accuracy: {accuracy * 100:.4f}")
    tensorboard_writer.add_scalar("Loss/Validation", loss, val_epoch)
    tensorboard_writer.add_scalar("Accuracy/Validation", accuracy, val_epoch)
    return accuracy

def main():
    """Put your training code for exercises 5.2 and 5.3 here"""

    data_path = "dataset/train"
    img_size = 224
    batch_size = 8
    num_workers = 0

    scale_factor = 4.0

    num_rows = int(img_size / scale_factor)
    num_cols = int(img_size / scale_factor)

    anchor_widths = [2, 4, 8, 16, 32, 64, 128, 256]
    aspect_ratios = [0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2, 3]

    agrid = anchor_grid.get_anchor_grid(
        num_rows,
        num_cols, 
        scale_factor,
        anchor_widths,
        aspect_ratios
    )

    model = MmpNet(len(anchor_widths) ,len(aspect_ratios))
    dataloader = dataset.get_dataloader(data_path, img_size, batch_size, num_workers, agrid, False)

    loss_func, optimizer = a2.get_criterion_optimizer(model)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    tensorboard_writer = get_tensorboard_writer("MmpNet-A5")

    epochs = 10
    for i in range(epochs):
        curr_epoch = i
        train_epoch(model, dataloader, loss_func, optimizer, device)
        val_epoch(i, model, dataloader, device, tensorboard_writer)

if __name__ == "__main__":
    main()
