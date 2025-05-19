from typing import Sequence
from PIL import Image
import random
import numpy as np
import json
import torch
import torchvision

def build_batch(paths: Sequence[str], transform=None) -> torch.Tensor:
    """Exercise 1.1

    @param paths: A sequence (e.g. list) of strings, each specifying the location of an image file.
    @param transform: One or multiple image transformations for augmenting the batch images.
    @return: Returns one single tensor that contains every image.
    """

    images = list()
    transforms = list()
    
    if transform:
        transforms.append(transform)
    
    if not isinstance(transform, torchvision.transforms.Resize):
        transforms.append(torchvision.transforms.Resize((224, 224)))

    transforms.append(torchvision.transforms.ToTensor())
    transforms.append(torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))

    tfm = torchvision.transforms.Compose(transforms)

    for path in paths:
        img = Image.open(path)
        img_tensor = tfm(img)
        if not isinstance(img_tensor, torch.Tensor):
            img_tensor = torch.tensor(img_tensor)
        images.append(img_tensor)

    img_batch = torch.stack(images)
    return img_batch


def get_model() -> torch.nn.Module:
    """Exercise 1.2

    @return: Returns a neural network, initialised with pretrained weights.
    """
    model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
    model.eval()
    
    return model

def print_predictions(title, model, img_batch, img_paths, classmap):
    
    print(8*"*")
    print(f"{title}")
    print(8*"*")

    output = model(img_batch)
    top_pred = output.argmax(dim=1)

    for i in range(len(img_paths)):
        expecting = img_paths[i].removeprefix("images/").removesuffix(".jpg").capitalize()
        print(f"Prediction for {expecting}: {classmap[top_pred[i]]}")

def main():
    """Exercise 1.3

    Put all your code for exercise 1.3 here.
    """

    img_paths = [
        "images/golden retriever.jpg",
        "images/koala.jpg",
        "images/pacifier.jpg",
        "images/rubber ducks.jpg",
        "images/rubber duck sculpture.jpg",
        "images/shoehorn.jpg",
        "images/zoo.jpg",
    ]

    # (a)
    def_img_batch = build_batch(img_paths)

    # (b)
    img_size = random.randint(128, 720)
    resized_img_batch = build_batch(img_paths, torchvision.transforms.Resize((img_size, img_size)))

    # (c)
    vertically_flipped_img_batch = build_batch(img_paths, torchvision.transforms.RandomVerticalFlip(1.0))

    model = get_model()

    with open("imagenet_classes.json") as f:
        classmap = json.load(f)

    print_predictions("Default Transforms", model, def_img_batch, img_paths, classmap)
    print_predictions(f"Resized Images to {img_size} x {img_size}", model, resized_img_batch, img_paths, classmap)
    print_predictions("Flipped Images Vertically", model, vertically_flipped_img_batch, img_paths, classmap)

if __name__ == "__main__":
    main()
