from typing import List, Tuple
import torch
import gc

from datetime import datetime
from ..a4.anchor_grid import get_anchor_grid
from ..a5.main import get_criterion_optimizer, get_tensorboard_writer, train_epoch
from ..a5.model import MmpNet
from ..a6.main import evaluate
from ..a7.dataset import get_dataloader, get_blur_transformation, get_color_transformation, get_grayscale_transformation, get_horizontal_flip_transformation, get_rotation_transformation

# Only import tensorboard if it is installed
try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None

curr_epoch = 0

curr_eval_epoch = 0
curr_eval_batch = 0

nms_threshold = 0.3

def main():

    use_negative_mining = True
    epochs = 50
    scale_factor = 8.0
    learn_rate = 0.001
    train_data_path = "new_dataset/train"
    eval_data_path = "new_dataset/val"
    anchor_widths = [4, 8, 16, 32, 64, 128, 224]
    aspect_ratios = [0.1, 0.25, 0.5, 1.0, 1.5, 2.0]
    img_size = 224
    batch_size = 32
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

    augmentation_combinations = [
        ("brightness_flip", [get_color_transformation(brightness=1.0), get_horizontal_flip_transformation()]),
        ("grayscale_rotated", [get_grayscale_transformation(), get_rotation_transformation(45)]),
        ("blur_rotated", [get_blur_transformation(), get_rotation_transformation(45)]),
        ("grayscale_brightness", [get_grayscale_transformation(), get_color_transformation(brightness=2.0)])
    ]

    results = {}

    for combinations in augmentation_combinations:
        name = combinations[0]
        augments = combinations[1]

        model = MmpNet(len(anchor_widths), len(aspect_ratios), num_rows, num_cols)
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        loss_func, optimizer = get_criterion_optimizer(model, learn_rate, device)

        augmented_training_loader = get_dataloader(train_data_path, img_size, batch_size, num_workers, anchor_grid, False, augmentations=augments)
        eval_loader = get_dataloader(eval_data_path, img_size, 1, num_workers, anchor_grid, True)

        train(epochs, model, loss_func, optimizer, device, augmented_training_loader, use_negative_mining, evaluate=False, augments=name)
        
        ap = get_average_precision(model, eval_loader, device, augments=name)
        
        results[name] = ap

        # Free model and dataset from memory
        del model
        del augmented_training_loader
        del eval_loader
        del optimizer
        torch.cuda.empty_cache()
        gc.collect()


def train(epochs, model, loss_func, optimizer, device, loader, negative_mining = True, evaluate = False, augments = ""):

    global curr_eval_epoch

    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M")

    tensorboard_writer = get_tensorboard_writer(f"a7_training_{augments}_{timestamp}")

    anchor_grid = loader.dataset.anchor_grid

    try:
        for epoch in range(epochs):
            curr_eval_epoch
            train_epoch(model, loader, loss_func, optimizer, device, negative_mining)
            if evaluate:
                ap = evaluate(model, loader, device, None, anchor_grid)
                tensorboard_writer.add_scalar("Precision/Epoch", ap, epoch)
                print(f"Precision on epoch {epoch}: {ap}")
            # Save model's weights every 20 epochs
            if (epoch+1) % 25 == 0:
                torch.save(model.state_dict(), f"a7_e{epoch}_{timestamp}.pth")
    finally:
        tensorboard_writer.close() # Close writer even on failure

def get_average_precision(model, loader, device, augments = ""):

    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M")

    data_path = loader.dataset.path_to_data.removeprefix("new_dataset/").removesuffix("/")

    tensorboard_writer = get_tensorboard_writer(f"a7_evaluate_dataset-{data_path}_{augments}_{timestamp}")

    anchor_grid = loader.dataset.anchor_grid
    try:
        ap = evaluate(model, loader, device, tensorboard_writer, anchor_grid)
    finally:
        tensorboard_writer.close()
    return ap

if __name__ == "__main__":
    main()
