from typing import List, Sequence, Tuple

from utils.annotation import AnnotationRect
from utils.label_grid import iou

def non_maximum_suppression(
    boxes_scores: Sequence[Tuple[AnnotationRect, float]], threshold: float
) -> List[Tuple[AnnotationRect, float]]:
    """
    @param boxes_scores: Sequence of tuples of annotations and scores
    @param threshold: Threshold for NMS

    @return: A list of tuples of the remaining boxes after NMS together with their scores
    """
    result = set()
    sorted_boxes_scores = sorted(boxes_scores, key = lambda x: x[1], reverse=True)
    while len(sorted_boxes_scores) > 0:
        M = sorted_boxes_scores.pop(0)
        result.add(M)
        box = M[0]

        sorted_boxes_scores = [
            bs for bs in sorted_boxes_scores if iou(box, bs[0]) <= threshold
        ]
    
    return list(result)
