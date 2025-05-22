from typing import List
from pathlib import Path
import numpy as np
import os
from PIL import Image, ImageDraw

class AnnotationRect:
    """Exercise 3.1"""

    def __init__(self, x1, y1, x2, y2):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2

    def area(self):
        height = abs(self.y1 - self.y2)
        width = abs(self.x1 - self.x2)
        return height * width
    
    def resized(self, scale_x, scale_y):
        return AnnotationRect(
            self.x1 * scale_x,
            self.y1 * scale_y,
            self.x2 * scale_x,
            self.y2 * scale_y
        )

    def __array__(self) -> np.ndarray:
        return np.array([self.x1, self.y1, self.x2, self.y2], dtype=np.int64)

    @staticmethod
    def fromarray(arr: np.ndarray):
        return AnnotationRect(arr[0], arr[1], arr[2], arr[3])

def read_groundtruth_file(path: str) -> List[AnnotationRect]:
    """Exercise 3.1b"""

    annotationRects = list()
    with open(path) as f:
        for line in f.readlines():
            tokens = line.split(" ")
            values = np.array(list(float(t) for t in tokens))
            annotationRects.append(AnnotationRect.fromarray(values))
    return annotationRects

# put your solution for exercise 3.1c wherever you deem it right

def draw_img_with_max_annotation_count(dir_path):
    img_path = get_img_with_max_annotation_count(dir_path)
    draw_img(img_path)

def get_img_with_max_annotation_count(path: str) -> str:
    groundtruthFiles = list(gt for gt in os.listdir(path) if gt.endswith("gt_data.txt"))
    result = (None, 0)
    for gt_filename in groundtruthFiles:
        gt_file_path = path + gt_filename if path.endswith("/") else path + "/" + gt_filename
        annotations = read_groundtruth_file(gt_file_path)
        annotionCount = len(annotations)
        if result[1] < annotionCount:
            result = (gt_file_path.replace(".gt_data.txt", ".jpg"), annotionCount)
    return result[0]

def draw_annotation(img_draw: np.array, ann: np.array, color = "blue", thickness: int = 3):
    img_draw.rectangle([(int(ann[0]), int(ann[1])), (int(ann[2]), int(ann[3]))], outline=color, width=thickness)

def draw_img(path: str):

    img = Image.open(path)
    np_img = np.array(img)

    groundtruthPath = path.replace(".jpg", ".gt_data.txt")
    annotations = read_groundtruth_file(groundtruthPath)

    for annotation in annotations:
        img_draw = ImageDraw.Draw(img)
        draw_annotation(img_draw, annotation.__array__())

    img.show()

"""Uncomment to draw image for exercise 3.1c"""
#dataset_dir = f"{Path.cwd().parent.parent}/dataset/train"
#draw_img_with_max_annotation_count(dataset_dir)