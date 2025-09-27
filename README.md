# Human Detection with Transfer Learning (PyTorch + MobileNetV2)

This project demonstrates how to use transfer learning with PyTorch’s `mobilenet_v2` model to perform human detection with bounding boxes. 
Unlike standard image classification, this work extends into object detection by localizing humans in an image rather than just predicting whether one exists.

---

## ✨ Project Overview
The project was developed as part of a university course on Computer Vision. It is structured as a sequence of assignments (`a1` → `a8`), each building on the previous to gradually develop the full detection pipeline:

- Understanding image preprocessing and data pipelines 
- Using MobileNetV2 as a feature extractor 
- Fine-tuning with transfer learning on a human detection dataset 
- Implementing bounding box regression in addition to classification 
- Training and evaluating the detector 
- Visualizing results with bounding boxes drawn over detections 

The final result is a compact detector that can identify and localize humans in images.

---

## 🛠️ Technologies Used
- [Python 3.x](https://www.python.org/) 
- [PyTorch](https://pytorch.org/) 
- [torchvision](https://pytorch.org/vision/stable/index.html) 
- [MobileNetV2](https://pytorch.org/vision/stable/models/generated/torchvision.models.mobilenet_v2.html) as backbone 

---

# Example Image Dataset

This repository contains a small example dataset of images along with bounding box annotations, intended for demonstration, testing, and educational purposes.

## Dataset Source

The example images included in this repository are from [Pexels.com](https://www.pexels.com/) and are provided here for demonstration purposes only. 

> ⚠️ Note: Pexels images are free to use, modify, and share, but redistribution of large numbers of original images is not allowed. This repository includes only a few sample images.

# Dataset Annotation Structure

This repository contains images along with their corresponding bounding box annotations for object detection tasks. Each image has a separate annotation file that describes its bounding boxes.

## Annotation Format

- Each image `img.jpg` has a corresponding ground truth file `img.gt_data.txt`.
- Each line in the `.gt_data.txt` file represents one bounding box in the image.
- Bounding box coordinates are stored as:

Where:

- `x1, y1` = coordinates of the bottom-left corner of the bounding box.

- `x2, y2` = coordinates of the top-right corner of the bounding box.

- All values are floats, e.g., `34.0 45.0 120.0 200.0`.

### Example

For an image `cat.jpg`, the annotation file `cat.gt_data.txt` might look like this:

34.0, 45.0, 120.0, 200.0

150.0, 60.0, 220.0, 180.0

- In this example, the image has two bounding boxes, each line representing one rectangle.

## Notes

- Ensure that the annotation filename exactly matches the image filename, except for the `.gt_data.txt` extension.
- The coordinates are absolute pixel values relative to the original image size.
- This structure is compatible with most object detection pipelines after parsing.


## 🚀 Running the Project

### WIP
