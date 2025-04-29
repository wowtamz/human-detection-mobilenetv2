from typing import Sequence
from PIL import Image
import json
import torch
import torchvision
import numpy as np

def build_batch(paths: Sequence[str], transform=None) -> torch.Tensor:
    """Exercise 1.1

    @param paths: A sequence (e.g. list) of strings, each specifying the location of an image file.
    @param transform: One or multiple image transformations for augmenting the batch images.
    @return: Returns one single tensor that contains every image.
    """

    images = list()

    tfm = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        torchvision.transforms.Resize((224, 224))
    ])

    for path in paths:
        img = Image.open(path)
        img_tensor = transform(img) if transform else tfm(img)
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
    tfm_b = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        torchvision.transforms.Resize((512, 512))
    ])
    resized_img_batch = build_batch(img_paths, tfm_b)

    # (c)
    tfm_c = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        torchvision.transforms.Resize((224, 224)),
        torchvision.transforms.RandomVerticalFlip(1.0)
    ])
    vertically_flipped_img_batch = build_batch(img_paths, tfm_c)

    model = get_model()

    with open("imagenet_classes.json") as f:
        classmap = json.load(f)

    print(8*"*")
    print("Default transforms")
    print(8*"*")

    output = model(def_img_batch)
    top_pred = output.argmax(dim=1)

    for i in range(len(img_paths)):
        expecting = img_paths[i].removeprefix("images/").removesuffix(".jpg").capitalize()
        print(f"Prediction for {expecting}: {classmap[top_pred[i]]}")
    
    print(8*"*")
    print("Resized images to 512 x 512")
    print(8*"*")

    output = model(resized_img_batch)
    top_pred = output.argmax(dim=1)

    for i in range(len(img_paths)):
        expecting = img_paths[i].removeprefix("images/").removesuffix(".jpg").capitalize()
        print(f"Prediction for {expecting}: {classmap[top_pred[i]]}")

    print(8*"*")
    print("Flipped images vertically")
    print(8*"*")

    output = model(vertically_flipped_img_batch)
    top_pred = output.argmax(dim=1)

    for i in range(len(img_paths)):
        expecting = img_paths[i].removeprefix("images/").removesuffix(".jpg").capitalize()
        print(f"Prediction for {expecting}: {classmap[top_pred[i]]}")

if __name__ == "__main__":
    main()
