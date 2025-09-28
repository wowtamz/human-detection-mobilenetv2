# Object Detection with Transfer Learning (PyTorch + MobileNetV2)

This project demonstrates how to use transfer learning with PyTorch’s `mobilenet_v2` model as a backbone to perform human detection with bounding boxes. 
Unlike standard image classification, this work extends into object detection by localizing humans in an image rather than just predicting whether one exists.

---

## ✨ Project Overview
The project was developed as part of a university course on Computer Vision. It is structured as a sequence of assignments (`a1` → `a8`), each building on the previous to gradually develop the full detection pipeline:

- Understanding image preprocessing and data pipelines 
- Using MobileNetV2 as a feature extractor 
- Fine-tuning with transfer learning on a human detection dataset
- Training and evaluating the detector 
- Implementing bounding box regression in addition to classification 
- Visualizing results with bounding boxes drawn over detections

After completing the assignments, the code was refactored into a cleaner, modular structure.
The final result is a compact detector that can identify and localize humans in images.

---

## 🚀 Running the Project

This repository includes a `demo.py` script that demonstrates how to train and evaluate the model 
using the example dataset.

> ⚠️ **Important:**  
> The project is structured as a Python **module**.  
> You must run it with `python3 -m` for imports to work correctly.

### Prerequisites
- Python 3.8 or higher
- Pip (Python package manager)

### Clone the project

```bash
  git clone https://github.com/wowtamz/human-detection-mobilenetv2
```

### Go to the project directory

```bash
  cd human-detection-mobilenetv2
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run the Demo

```bash
python3 -m demo
```

By default, the demo script:

- Creates a `DataLoader` for the `test`, `train` and `validation` datasets from the `example_dataset` folder.

- Draws the ground truth labels on the `train` dataset images and stores the results in the `output/ground_truth` folder.

- Computes predictions of the untrained mode on the `test` set and stores the results in the `output` folder.

- Overfits the simple model to a single image from the `example_dataset/overfit` folder and stores the result in the `output` folder.

- Trains the `SimpleNet` model on the `train` set and computes its average precision on all datasets as well as its prediction for the `test` set and stores the results in the `output` folder.

- Trains the `BBRNet` model on the `train` set and computes its average precision on all datasets as well as its prediction for the `test` set and stores the results in the `output` folder.

- Prints the average precision metrics of both models to the console.

---

## 📊 Example Image Dataset

This repository contains a small example dataset of images along with bounding box annotations, intended for demonstration, testing, and educational purposes.

### Dataset Structure

The dataset is organized into three  main subfolders.
Each of these represents a separate dataset split:

- `test/` is reserved for final evaluation of the trained model.

- `train/` is used for training the model, 

- `val/` is used during training to validate performance and tune hyperparameters, 


```
example_dataset/
├── test/
│   ├── people.jpg
│   └── people.gt_data.txt
├── train/
│   ├── workers.jpg
│   ├── workers.gt_data.txt
│   ├── dog.jpg
│   └── dog.gt_data.txt
└── val/
    ├── horse.jpg
    └── horse.gt_data.txt
```

### Dataset Source

The example images included in this repository are from [Pexels.com](https://www.pexels.com/) and are provided here for demonstration purposes only. 

> ⚠️ Note: Pexels images are free to use, modify, and share, but redistribution of large numbers of original images is not allowed. This repository includes only a few sample images.

## 🔖 Dataset Annotation Structure

This repository contains images along with their corresponding bounding box annotations for object detection tasks. Each image has a separate annotation file that describes its bounding boxes.

### Annotation Format

- Each image `img.jpg` has a corresponding ground truth file `img.gt_data.txt`.
- Each line in the `.gt_data.txt` file represents one bounding box in the image.
- Bounding box coordinates are stored as: `x1 y1 x2 y2`

Where:

- `x1, y1` = coordinates of the bottom-left corner of the bounding box.

- `x2, y2` = coordinates of the top-right corner of the bounding box.

- All values are floats, e.g., `34.0 45.0 120.0 200.0`.

#### Example

For an image `cats.jpg`, the annotation file `cats.gt_data.txt` might look like this:

```
34.0 45.0 120.0 200.0
150.0 60.0 220.0 180.0
```
- In this example, the image has two bounding boxes, each line representing one rectangle.

### Notes

- Ensure that the annotation filename exactly matches the image filename, except for the `.gt_data.txt` extension.
- The coordinates are absolute pixel values relative to the original image size.
