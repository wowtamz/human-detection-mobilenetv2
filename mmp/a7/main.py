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
    epochs = 100
    scale_factor = 8.0
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

    augmentation_combinations = [
        ("brightness_flip", [get_color_transformation(brightness=1.0), get_horizontal_flip_transformation()]),
        ("grayscale_rotated", [get_grayscale_transformation(), get_rotation_transformation(45)]),
        ("blur_rotated", [get_blur_transformation(), get_rotation_transformation(45)]),
        ("grayscale_brightness", [get_grayscale_transformation(), get_color_transformation(brightness=2.0)])
    ]

    benchmarks = {}

    # Evaluate AP of pretrained models
    augmentation_combinations = [] # Comment this line out to allow training
    model_paths = [

    ]

    for path in model_paths:
        
        model = MmpNet(len(anchor_widths), len(aspect_ratios), num_rows, num_cols)
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        state_dict = torch.load(path, map_location=torch.device(device))
        model.load_state_dict(state_dict)
        eval_loader = get_dataloader(eval_data_path, img_size, 1, num_workers, anchor_grid, True)
        
        ap = get_average_precision(model, eval_loader, device, augments=name)
        
        benchmarks[path] = ap

         # Free model and dataset from memory
        del model
        del eval_loader
        del state_dict
        torch.cuda.empty_cache()
        gc.collect()
    
    # Traing models with augmentation

    for combinations in augmentation_combinations:
        name = combinations[0]
        augments = combinations[1]

        model = MmpNet(len(anchor_widths), len(aspect_ratios), num_rows, num_cols)
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        loss_func, optimizer = get_criterion_optimizer(model, learn_rate, device)

        augmented_training_loader = get_dataloader(train_data_path, img_size, batch_size, num_workers, anchor_grid, False, augmentations=augments)
        eval_loader = get_dataloader(eval_data_path, img_size, 1, num_workers, anchor_grid, True)
        
        train(epochs, model, loss_func, optimizer, device, augmented_training_loader, use_negative_mining, evaluate_training=True, augments=name)
        
        ap = get_average_precision(model, eval_loader, device, augments=name)
        
        benchmarks[name] = ap

        # Free model and dataset from memory
        del model
        del augmented_training_loader
        del eval_loader
        del optimizer
        torch.cuda.empty_cache()
        gc.collect()
    
    print(8*"-", "Benchmarks", 8*"-")
    for name, ap in benchmarks.items():
        print(f"Augmentation: {name.replace('_', ' ').title()}")
        print(f"Average precision: {ap}")
        print(16*"-")

def train(epochs, model, loss_func, optimizer, device, loader, negative_mining = True, evaluate_training = False, augments = ""):

    global curr_eval_epoch

    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M")

    if evaluate_training:
        tensorboard_writer = get_tensorboard_writer(f"a7_training_{augments}_{timestamp}")

    anchor_grid = loader.dataset.anchor_grid

    try:
        for epoch in range(epochs):
            curr_eval_epoch = epoch
            train_epoch(model, loader, loss_func, optimizer, device, negative_mining)
            # Save model's weights every 25 epochs
            if (epoch+1) % 25 == 0:
                torch.save(model.state_dict(), f"a7_e{epoch+1}_{augments}_{timestamp}.pth")
            
            # Evaluate model every 20 epochs
            if evaluate_training and (epoch + 1) % 20 == 0:
                ap = evaluate(model, loader, device, None, anchor_grid)
                tensorboard_writer.add_scalar("Precision/Epoch", ap, epoch)
                print(f"Precision on epoch {epoch}: {ap}")
    finally:
        if evaluate_training:
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
