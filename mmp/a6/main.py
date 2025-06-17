from typing import List, Tuple
import torch
import numpy as np

from ..a3.annotation import AnnotationRect
from ..a4.label_grid import iou
from ..a5.model import MmpNet
from ..a5.main import get_criterion_optimizer
from .nms import non_maximum_suppression

# Only import tensorboard if it is installed
try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None

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

    precisions = np.array()
    recalls = np.array()

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

def evaluate_test():  # feel free to change the arguments
    """Generates predictions on the provided test dataset.
    This function saves the predictions to a text file.

    You decide which arguments this function should receive
    """
    raise NotImplementedError()


def main():
    """Put the surrounding training code here. The code will probably look very similar to last assignment"""
    raise NotImplementedError()


if __name__ == "__main__":
    main()
