import torch
import gc

from datetime import datetime
from ..a4.anchor_grid import get_anchor_grid
from ..a5.main import get_criterion_optimizer
from ..a5.model import MmpNet
from ..a6.main import evaluate, batch_inference
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

benchmarks = {}

use_negative_mining = True
scale_factor = 8.0
learn_rate = 0.001
train_data_path = "new_dataset/train"
eval_data_path = "new_dataset/val"
anchor_widths = [16, 32, 64, 128, 224]
aspect_ratios = [0.25, 0.5, 1.0, 1.5, 2.0]
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

def main():

    use_negative_mining = True
    epochs = 100
    scale_factor = 8.0
    learn_rate = 0.001
    train_data_path = "new_dataset/train"
    eval_data_path = "new_dataset/val"
    anchor_widths = [16, 32, 64, 128, 224]
    aspect_ratios = [0.25, 0.5, 1.0, 1.5, 2.0]
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

    # Evaluate pretrained models

    for tup in [
        #("without_bbr", "a8_without_bbr_e100__2025-07-23-14:37.pth"),
        #("with_bbr", "a8_with_bbr_e100__2025-07-22-10:17.pth"),
        #("exp1", "exp_seq_aug_general_to_specific_2025-07-24-19:57.pth"),
        #("exp2", "exp_seq_aug_specific_to_general_2025-07-25-21:41.pth"),
        #("exp3", "exp_seq_aug_more_rounds_2025-07-26-05:50.pth"),
        #("exp4", "exp_seq_aug_single_round_2025-07-26-19:09.pth"),
        #
    ]:
        name = tup[0]
        path = tup[1]

        allow_bbr = False if "without_bbr" in name else True
        model = MmpNet(len(anchor_widths), len(aspect_ratios), num_rows, num_cols, use_bbr=allow_bbr)
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
    
    # Exercise 8.1 
    # Train model with and without BBR
    bbrs = [True,
            False,
            ]

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
        del training_loader
        del eval_loader
        del optimizer
        torch.cuda.empty_cache()
        gc.collect()

    # Exercise 8.2 Performance Tuning

    augmentations = [
        [get_grayscale_transformation()],
        [get_blur_transformation()],
        [],
        [get_color_transformation(brightness=2.0), get_horizontal_flip_transformation()],
        [get_color_transformation(brightness=2.0), get_rotation_transformation(25)],
    ]

    # Experiment 1 - Sequential Augment Training (general -> specific features)

    
    run_experiment(
        "seq_aug_general_to_specific",
        augment_list=augmentations,
        total_epochs=250,
        rounds=2
    )    

    # Experiment 2 - Sequential Augment Training (specific -> general)

    run_experiment(
        "seq_aug_specific_to_general",
        augment_list=augmentations[::-1], # Reversed augmentations list
        total_epochs=250,
        rounds=2
    )

    # Experiment 3 - Sequential Augment Training (more rounds)

    run_experiment(
        "seq_aug_more_rounds",
        augment_list=augmentations,
        total_epochs=250,
        rounds=10
    )

    # Experiment 4 - Sequential Augment Training (single round)

    run_experiment(
        "seq_aug_single_round",
        augment_list=augmentations,
        total_epochs=250,
        rounds=1
    )

    # Experiment 5 - Randomized Augment

    run_experiment(
        "rand_aug",
        augment_list=augmentations,
        total_epochs=250,
        rounds=5,
        randomize=True
    )
    
    print(8*"-", "Benchmarks", 8*"-")
    for name, ap in benchmarks.items():
        print(f"Augmentation: {name.replace('_', ' ').title()}")
        print(f"Average precision: {ap}")
        print(16*"-")

def run_experiment(exp_name, augment_list, total_epochs, rounds=1, randomize=False):

    timestamp = datetime.now().strftime("%Y-%m-%d-%H:%M")
    print(f"Model training experiment: {exp_name}")

    using_bbr = "no_bbr" not in exp_name

    model = MmpNet(len(anchor_widths), len(aspect_ratios), num_rows, num_cols, use_bbr=using_bbr)
    learn_rate = 0.001
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    loss_func, optimizer = get_criterion_optimizer(model, learn_rate, device)

    assert total_epochs % rounds == 0, "Total epochs must be divisable by round count"
    assert total_epochs % (rounds * len(augment_list)) == 0, "Total epochs must be divisable by rounds * augment count!"
    epochs_per_augment = int(total_epochs / rounds / len(augment_list))

    for r in range(rounds):

        rand_epochs = split_number(total_epochs / rounds, len(augment_list)) if randomize else None

        for i, augments in enumerate(augment_list):

            if randomize:
                rand_epochs_per_augment = rand_epochs[i]
            
            augmented_training_loader = get_dataloader(train_data_path, img_size, batch_size, num_workers, anchor_grid, False, augmentations=augments)
            train(rand_epochs_per_augment if randomize else epochs_per_augment, model, loss_func, optimizer, device, augmented_training_loader, use_negative_mining, evaluate_training=False, tag=exp_name)
            
            # Free dataset from memory
            del augmented_training_loader
            torch.cuda.empty_cache()
            gc.collect()
    
    torch.save(model.state_dict(), f"exp_{exp_name}_{timestamp}.pth")
    eval_loader = get_dataloader(eval_data_path, img_size, 1, num_workers, anchor_grid, True)
    ap = get_average_precision(model, eval_loader, device, augments="performance_tuning", tag=f"exp_{exp_name}_e{total_epochs}")
    evaluate_experiment(tag, model, eval_loader, device, anchor_grid, timestamp)
    benchmarks[f"exp_{exp_name}"] = ap

    # Free model and eval dataset from memory
    del model
    del eval_loader
    del optimizer
    torch.cuda.empty_cache()
    gc.collect()


def split_number(total, parts):
    # Choose (parts - 1) unique split points between 1 and total - 1
    split_points = sorted(random.sample(range(1, total), parts - 1))
    
    # Add 0 and total as start and end boundaries
    split_points = [0] + split_points + [total]
    
    # Compute differences between consecutive split points
    return [split_points[i+1] - split_points[i] for i in range(parts)]

def evaluate_experiment(tag, model, data_loader, device, anchor_grid, timestamp):
    """Generates predictions on the provided test dataset.
    This function saves the predictions to a text file.

    You decide which arguments this function should receive
    """

    lines = []
    
    model = model.to(device)
    model.eval()

    count = 0

    dataset = data_loader.dataset
    
    for batch in data_loader:
        images, labels, ids = batch
        print(f"Evaluating img {count}")
        inference = batch_inference(model, images, device, anchor_grid)

        for i, img_id in enumerate(ids):

            prediction = inference[i]

            for box_score in prediction:
                rect = box_score[0]
                
                score = box_score[1]
                lines.append(f"{img_id} {int(rect.x1)} {int(rect.y1)} {int(rect.x2)} {int(rect.y2)} {score:.14f}\n")
        
        count += 1
        
    with open(f"mmp/a8/exp_{tag}_{timestamp}.txt", "w") as f:
        f.writelines(lines)

if __name__ == "__main__":
    main()
