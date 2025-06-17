from typing import List, Tuple
import torch
import numpy as np

from ..a3.annotation import AnnotationRect
from ..a4.label_grid import iou
from ..a5.model import MmpNet
from .nms import non_maximum_suppression

def batch_inference(
    model: MmpNet, images: torch.Tensor, device: torch.device, anchor_grid: np.ndarray
) -> List[List[Tuple[AnnotationRect, float]]]:
    
    result = []

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

def evaluate() -> float:  # feel free to change the arguments
    """Evaluates a specified model on the whole validation dataset.

    @return: AP for the validation set as a float.

    You decide which arguments this function should receive
    """
    raise NotImplementedError()


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
