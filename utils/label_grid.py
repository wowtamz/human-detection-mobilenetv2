from typing import Sequence, Tuple
from utils.annotation import AnnotationRect
import numpy as np

def iou(rect1: AnnotationRect, rect2: AnnotationRect) -> float:
    intersect_rect_x1 = max(rect1.x1, rect2.x1)
    intersect_rect_y1 = max(rect1.y1, rect2.y1)
    intersect_rect_x2 = min(rect1.x2, rect2.x2)
    intersect_rect_y2 = min(rect1.y2, rect2.y2)
    
    if (intersect_rect_x1 >= intersect_rect_x2 or intersect_rect_y1 >= intersect_rect_y2):
        return 0.0

    intersect_rect = AnnotationRect(intersect_rect_x1, intersect_rect_y1, intersect_rect_x2, intersect_rect_y2)
    union_area = rect1.area() + rect2.area() - intersect_rect.area()
    
    return intersect_rect.area() / union_area


def get_label_grid(
    anchor_grid: np.ndarray, gts: Sequence[AnnotationRect], min_iou: float
) -> tuple[np.ndarray, ...]:
    
    sizes, ratios, rows, cols, _ = anchor_grid.shape

    grid_size = sizes * ratios * rows * cols

    grid = np.zeros(grid_size, dtype=bool) # Preallocate flat grid memory

    anchor_grid_flat = anchor_grid.reshape(-1, 4)
    
    for i in range(grid_size):
        for gt in gts:
            rect = anchor_grid_flat[i]
            if iou(AnnotationRect.fromarray(rect), gt) >= min_iou:
                grid[i] = True
                continue
    
    grid = grid.reshape((sizes, ratios, rows, cols))

    return grid

def get_matching_rects(anch_grid, label_grid):
    rects = []
    sizes, ratios, rows, cols, _ = anch_grid.shape
    for size in range(sizes):
        for ratio in range(ratios):
            for row in range(rows):
                for col in range(cols):
                    if label_grid[size, ratio, row, col]:
                        rect = AnnotationRect.fromarray(anch_grid[size, ratio, row, col])
                        rects.append(rect)
    return rects

def get_pred_rects_scores(anch_grid, label_grid, detection_threshold = 0.5) -> list[Tuple[AnnotationRect, float]]:
    rects_scores = []
    sizes, ratios, rows, cols, _ = anch_grid.shape
    for size in range(sizes):
        for ratio in range(ratios):
            for row in range(rows):
                for col in range(cols):
                    score = label_grid[size, ratio, row, col]
                    if score >= detection_threshold:
                        rect = AnnotationRect.fromarray(anch_grid[size, ratio, row, col])
                        rects_scores.append((rect, score))
    return rects_scores