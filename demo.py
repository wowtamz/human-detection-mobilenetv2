import torch

from models.simplenet import SimpleNet
from utils.anchor_grid import get_anchor_grid
from utils.dataset import get_dataloader
from training.train import train_model

scale_factor = 16.0
learn_rate = 0.001
train_data_path = "dataset/train"
eval_data_path = "dataset/val"
anchor_widths = [8, 32, 64, 128, 256]
aspect_ratios = [0.5, 1.0, 2.0]

img_size = 224

batch_size = 8
num_workers = 0

num_rows = int(img_size / scale_factor)
num_cols = int(img_size / scale_factor)

anchor_grid = get_anchor_grid(
    num_rows,
    num_cols,
    scale_factor,
    anchor_widths,
    aspect_ratios
)

def main():
    epochs = input("Please specify the amount of epochs to train")

    epochs = int(epochs)

    model = SimpleNet(len(anchor_widths), len(aspect_ratios), num_rows, num_cols)
    
    training_loader = get_dataloader(train_data_path, img_size, 32, num_workers, anchor_grid, min_iou=0.5, is_test=False)

    train_model(model, training_loader, learn_rate, epochs, save=True)

if __name__ == "__main__":
    main()