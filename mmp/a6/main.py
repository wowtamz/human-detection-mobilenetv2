from typing import List, Tuple
import torch
import numpy as np

from ..a3.annotation import AnnotationRect
from ..a4.label_grid import iou
from ..a4.anchor_grid import get_anchor_grid
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

def batch_inference(
    model: MmpNet, images: torch.Tensor, device: torch.device, anchor_grid: np.ndarray
) -> List[List[Tuple[AnnotationRect, float]]]:

    result = []
    images = images.to(device)

    for i in images:
        boxes_scores = []
        predicition = model(i)
        widths, ratios, rows, cols = predicition.shape
        for w in range(widths):
            for a in range(ratios):
                for r in range(rows):
                    for c in range(cols):
                        pred_box_array = predicition[w][a][r][c]
                        pred_rect = AnnotationRect.fromarray(pred_box_array)

                        anchor_box_array = anchor_grid[w][a][r][c]
                        anchor_rect = AnnotationRect.fromarray(anchor_box_array)
                        score = iou(pred_rect, anchor_rect)
                        boxes_scores.append((pred_rect, score))
        filtered = non_maximum_suppression(boxes_scores)
        result.append(filtered)

    return result

def evaluate(model, loader, device, tensorboard_writer, anchor_grid) -> float:  # feel free to change the arguments
    """Evaluates a specified model on the whole validation dataset.

    @return: AP for the validation set as a float.

    You decide which arguments this function should receive
    """
    
    model = model.to(device)
    model.eval()

    precisions = np.array([])
    recalls = np.array([])

    with torch.no_grad():
        for i, data in enumerate(loader):
            images, labels, ids = data
            
            inference = batch_inference(model, images, device, anchor_grid)
            tp = torch.sum(inference == labels)
            fp = torch.sum(inference == True and labels == False)
            fn = torch.sum(inference == False and labels == True)

            prec = tp / (tp + fp)
            rec = tp / (tp + fn)
            precisions.append(prec)
            recalls.append(rec)
            print(f"Evaluating batch {i}, Precision ({prec}), Recall ({rec})")

    avg_precision = np.average(precisions)
    avg_recall = np.average(recalls)

    if tensorboard_writer:
        tensorboard_writer.add_scalar("Precision/Recall", avg_precision, avg_recall)
    
    return avg_precision

def evaluate_test(model, data_loader, device, anchor_grid):  # feel free to change the arguments
    """Generates predictions on the provided test dataset.
    This function saves the predictions to a text file.

    You decide which arguments this function should receive
    """
    # like model_output.txt

    lines = []

    model.eval()
    
    for batch in data_loader:
        images, labels, ids = batch
        images = images.to(device)
        labels = labels.to(device)

        prediction = model(images)

        widths, aspect_ratios, rows, cols = anchor_grid.shape()

        rects = []

        for w in range(widths):
            for a in range(aspect_ratios):
                for r in range(rows):
                    for c in range(cols):
                        if prediction[w][a][r][c] == True:
                            rect_array = anchor_grid[w][a][r][c]
                            rects.append(AnnotationRect.fromarray(rect_array))
        
        for rect in rects:
            lines.append(f"{ids} {rect.x1} {rect.y1} {rect.x2} {rect.y2}\n")
        
    with open("mmp/a6/eval_test.txt", "w") as f:
        f.writelines(lines)
        f.close()

def main():
    """Put the surrounding training code here. The code will probably look very similar to last assignment"""
    epochs = 20
    scale_factor = 8.0
    learn_rate = 0.02
    train_data_path = "dataset/train"
    eval_data_path = "dataset/val"
    anchor_widths = [2, 4, 8, 16, 32, 64, 128, 256]
    aspect_ratios = [0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]
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
            train_epoch(model, train_data_loader, loss_func, optimizer, device)
            prec_epoch = evaluate(model, train_data_loader, device, None, anchor_grid)
            tensorboard_writer.add_scalar("Precision/Epoch", prec_epoch, i)
            print(f"Precision on epoch {i}: {prec_epoch}")

        avg_precision = evaluate(model, eval_data_loader, device, tensorboard_writer, anchor_grid)
        print(f"Average precision on validation set: {avg_precision}")

    finally:
        tensorboard_writer.close()

    
if __name__ == "__main__":
    main()
