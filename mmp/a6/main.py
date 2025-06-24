from typing import List, Tuple
import torch
import numpy as np

from ..a3.annotation import AnnotationRect
from ..a4.anchor_grid import get_anchor_grid
from ..a4.label_grid import get_label_grid
from ..a4 import dataset
from ..a4.dataset import get_dataloader
from ..a5.model import MmpNet
from ..a5.main import get_criterion_optimizer, get_tensorboard_writer, train_epoch
from .nms import non_maximum_suppression

# Only import tensorboard if it is installed
try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None

curr_epoch = 0

curr_eval_epoch = 0
curr_eval_batch = 0

nms_threshold = 0.3

def batch_inference(
    model: MmpNet, images: torch.Tensor, device: torch.device, anchor_grid: np.ndarray
) -> List[List[Tuple[AnnotationRect, float]]]:
    
    global curr_eval_epoch
    global curr_eval_batch
    global nms_threshold

    result = []
    images = images.to(device)

    prediction = model(images)
    batch_size, widths, ratios, rows, cols = prediction.shape

    i_flat_anchor = anchor_grid.reshape(-1, 4)
    
    for i in range(batch_size):
        print(f"evaluating e:{curr_eval_epoch}/b:{curr_eval_batch}/img:{i}")
        i_flat_pred = prediction[i].reshape(-1)
    
        boxes_scores = []

        for pred_arr, anchor_arr in zip(i_flat_pred, i_flat_anchor):
            score = pred_arr
            anchor_rect = AnnotationRect.fromarray(anchor_arr)
            boxes_scores.append((anchor_rect, score))
        
        filtered = non_maximum_suppression(boxes_scores, nms_threshold)
        result.append(filtered)

    return result

def evaluate(model, loader, device, tensorboard_writer, anchor_grid) -> float:  # feel free to change the arguments
    """Evaluates a specified model on the whole validation dataset.

    @return: AP for the validation set as a float.

    You decide which arguments this function should receive
    """

    global curr_eval_batch
    
    model = model.to(device)
    model.eval()

    precisions = []
    recalls = []

    with torch.no_grad():
        for i, data in enumerate(loader):
            curr_eval_batch = i
            images, labels, ids = data
            
            inference = batch_inference(model, images, device, anchor_grid)
            pred_grids = [get_label_grid(anchor_grid, [box_score[0] for box_score in tups], 1.0) for tups in inference]
            pred_grids = [torch.from_numpy(p) for p in pred_grids] # Converts the label grids from numpy arrays to tensors
            prediction = torch.stack(pred_grids, dim=0) # Converts the list of label grids to a tensor
            tp = torch.sum(prediction == labels)
            fp = torch.sum((prediction == True) & (labels == False)) # Must use bitwise on tensors
            fn = torch.sum((prediction == False) & (labels == True)) # Must use bitwise on tensors

            prec = tp / (tp + fp)
            rec = tp / (tp + fn)
            precisions.append(prec)
            recalls.append(rec)
            print(f"Evaluated batch {i}, Precision ({prec}), Recall ({rec})")

    avg_precision = np.average(np.array(precisions))
    avg_recall = np.average(np.array(recalls))

    if tensorboard_writer:
        tensorboard_writer.add_scalar("Precision/Recall", avg_precision, avg_recall)
    
    return avg_precision

def evaluate_test(model, data_loader, device, anchor_grid):  # feel free to change the arguments
    """Generates predictions on the provided test dataset.
    This function saves the predictions to a text file.

    You decide which arguments this function should receive
    """

    lines = []

    model.eval()
    
    for batch in data_loader:
        images, labels, ids = batch

        predictions = batch_inference(model, images, device, anchor_grid)

        i_flat_anchor = anchor_grid.reshape(-1, 4)
                              
        for i, img_id in enumerate(ids):
            i_flat_pred = predictions[i].reshape(-1)
            
            for pred_arr, anchor_arr in zip(i_flat_pred, i_flat_anchor):
                score = pred_arr
                rect = AnnotationRect.fromarray(anchor_arr)
                lines.append(f"{img_id} {rect.x1} {rect.y1} {rect.x2} {rect.y2} {score}\n")
        
    with open("mmp/a6/eval_test.txt", "w") as f:
        f.writelines(lines)
        f.close()

def main():
    """Put the surrounding training code here. The code will probably look very similar to last assignment"""
    global curr_eval_epoch
    global nms_threshold

    use_negative_mining = True
    epochs = 10
    scale_factor = 8.0
    learn_rate = 0.02
    train_data_path = "dataset/train"
    eval_data_path = "dataset/val"
    anchor_widths = [8, 16, 32, 64, 128]
    aspect_ratios = [0.25, 0.5, 0.75, 1.0,2.0]
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

    model = MmpNet(len(anchor_widths), len(aspect_ratios), num_rows, num_cols)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    train_data_loader = get_dataloader(train_data_path, img_size, batch_size, num_workers, anchor_grid, False)
    eval_data_loader = get_dataloader(eval_data_path, img_size, 1, num_workers, anchor_grid, True)

    loss_func, optimizer = get_criterion_optimizer(model, learn_rate)

    tensorboard_writer = get_tensorboard_writer("Assignment 6.3")

    try:
        for i in range(epochs):
            curr_eval_epoch = i
            train_epoch(model, train_data_loader, loss_func, optimizer, device, use_negative_mining)
            prec_epoch = evaluate(model, train_data_loader, device, None, anchor_grid)
            tensorboard_writer.add_scalar("Precision/Epoch", prec_epoch, i)
            print(f"Precision on epoch {i}: {prec_epoch}")

        for confidence in np.arange(0.0, 1.0, 0.01):
            nms_threshold = confidence
            avg_precision = evaluate(model, eval_data_loader, device, tensorboard_writer, anchor_grid)
            print(f"Average precision on validation set with cut-off '{nms_threshold}': {avg_precision}")

    finally:
        tensorboard_writer.close()

    
if __name__ == "__main__":
    main()
