from typing import List, Sequence, Tuple

from ..a3.annotation import AnnotationRect, draw_annotation
from ..a4.label_grid import iou
from PIL import Image, ImageDraw

def non_maximum_suppression(
    boxes_scores: Sequence[Tuple[AnnotationRect, float]], threshold: float
) -> List[Tuple[AnnotationRect, float]]:
    """Exercise 6.1
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
            bs for bs in sorted_boxes_scores if iou(box, bs[0]) > threshold
        ]
    
    return list(result)

def exercise6_1_b():

    threshold = 0.3
    min_score = 0.5
    
    file_path = "mmp/a6/model_output.txt"
    output_dict: dict = dict()
    with open(file_path, "r") as f:
        for line in f.readlines():
            data = line.split(" ")
            img = data[0]
            x1 = float(data[1])
            y1 = float(data[2])
            x2 = float(data[3])
            y2 = float(data[4])
            rect: AnnotationRect = AnnotationRect(x1, y1, x2, y2)
            score: float = float(data[5])
            path: str = f"dataset/val/{img}.jpg"
            box_score: Tuple[AnnotationRect, float] = (rect, score)
            if path in output_dict.keys():
                output_dict[path].append(box_score)
            else:
                output_dict[path] = []

    for img_path in output_dict.keys():
        img_name = img_path.removeprefix("dataset/val/")
        box_scores: Sequence = output_dict[img_path]
        nms = non_maximum_suppression(box_scores, threshold)

        img = Image.open(img_path)
        draw = ImageDraw.Draw(img)

        for bs in nms:
            rect: AnnotationRect = bs[0]
            score = bs[1]
            if score > min_score:
                draw_annotation(draw, rect.__array__(), "red")
        
        img.save(f"mmp/a6/{img_name}")

if __name__ == "__main__":
    exercise6_1_b()