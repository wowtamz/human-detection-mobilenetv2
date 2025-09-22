from typing import List
import numpy as np


class AnnotationRect:
    """Exercise 3.1"""

    def __init__(self, x1, y1, x2, y2):
        raise NotImplementedError()

    def area(self):
        raise NotImplementedError()

    def __array__(self) -> np.ndarray:
        raise NotImplementedError()

    @staticmethod
    def fromarray(arr: np.ndarray):
        raise NotImplementedError()


def read_groundtruth_file(path: str) -> List[AnnotationRect]:
    """Exercise 3.1b"""
    raise NotImplementedError()


# put your solution for exercise 3.1c wherever you deem it right
