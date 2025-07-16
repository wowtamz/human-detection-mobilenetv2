import torch
import gc

from datetime import datetime
from ..a4.anchor_grid import get_anchor_grid
from ..a5.main import get_criterion_optimizer
from ..a5.model import MmpNet
from ..a6.main import evaluate
from ..a7.dataset import get_dataloader, get_blur_transformation, get_color_transformation, get_grayscale_transformation, get_horizontal_flip_transformation, get_rotation_transformation
from ..a7.main import train, get_average_precision

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
    epochs = 100
    scale_factor = 6.0
    learn_rate = 0.001
    train_data_path = "new_dataset/train"
    eval_data_path = "new_dataset/val"
    anchor_widths = [4, 8, 16, 32, 64, 128, 224]
    aspect_ratios = [0.1, 0.25, 0.5, 1.0, 1.5, 2.0]
    img_size = 224
    batch_size = 64
    num_workers = 4

    num_rows = int(img_size / scale_factor)
    num_cols = int(img_size / scale_factor)

    anchor_grid = get_anchor_grid(
        num_rows,
        num_cols,
        scale_factor,
        anchor_widths,
        aspect_ratios
    )

    benchmarks = {}

    # Exercise 8.1 
    # Train model with and without BBR
    bbrs = [False, True]

    for using_bbr in bbrs:
        w = "with" if using_bbr else "without"
        tag = f"a8_{w}_bbr"

        model = MmpNet(len(anchor_widths), len(aspect_ratios), num_rows, num_cols, use_bbr=using_bbr)
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        loss_func, optimizer = get_criterion_optimizer(model, learn_rate, device)
        
        training_loader = get_dataloader(train_data_path, img_size, batch_size, num_workers, anchor_grid, False)
        eval_loader = get_dataloader(eval_data_path, img_size, 1, num_workers, anchor_grid, True)
        
        train(epochs, model, loss_func, optimizer, device, training_loader, use_negative_mining, evaluate_training=False, tag=tag)
        
        ap = get_average_precision(model, eval_loader, device, augments="no_augs", tag=tag)

        benchmarks[tag] = ap

        del model
        del augmented_training_loader
        del eval_loader
        del optimizer
        torch.cuda.empty_cache()
        gc.collect()


    # Exercise 8.2 Performance Tuning
    # Custom model training

    model = MmpNet(len(anchor_widths), len(aspect_ratios), num_rows, num_cols, use_bbr=True)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    loss_func, optimizer = get_criterion_optimizer(model, learn_rate, device)

    rounds = 2

    for round in range(rounds):
        for augments in [
            [get_grayscale_transformation()],
            [get_blur_transformation()],
            [],
            [get_color_transformation(brightness=2.0), get_horizontal_flip_transformation()],
            [get_color_transformation(brightness=2.0), get_rotation_transformation(25)],
        ]:
            
            epochs_per_round = 25

            augmented_training_loader = get_dataloader(train_data_path, img_size, batch_size, num_workers, anchor_grid, False, augmentations=augments)
            eval_loader = get_dataloader(eval_data_path, img_size, 1, num_workers, anchor_grid, True)
            
            train(epochs_per_round, model, loss_func, optimizer, device, augmented_training_loader, use_negative_mining, evaluate_training=False, tag="performance_tuning")

            # Free model and dataset from memory
            del augmented_training_loader
            del eval_loader
            torch.cuda.empty_cache()
            gc.collect()
    
    ap = get_average_precision(model, eval_loader, device, augments="performance_tuning", tag=f"a8_e{rounds * epochs_per_round * 5}")
    benchmarks["performance_tuning"] = ap
    
    print(8*"-", "Benchmarks", 8*"-")
    for name, ap in benchmarks.items():
        print(f"Augmentation: {name.replace('_', ' ').title()}")
        print(f"Average precision: {ap}")
        print(16*"-")


if __name__ == "__main__":
    main()