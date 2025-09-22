from typing import List, Sequence, Tuple

from ..a3.annotation import AnnotationRect


def non_maximum_suppression(
    boxes_scores: Sequence[Tuple[AnnotationRect, float]], threshold: float
) -> List[Tuple[AnnotationRect, float]]:
    """Exercise 6.1
    @param boxes_scores: Sequence of tuples of annotations and scores
    @param threshold: Threshold for NMS

    @return: A list of tuples of the remaining boxes after NMS together with their scores
    """
    raise NotImplementedError()
