import torch

from torch import Tensor
from models.simplenet import SimpleNet
from utils.anchor_grid import get_anchor_grid
from utils.label_grid import get_pred_rects
from utils.dataset import get_dataloader
from training.train import train_model
from utils import annotation

from torchvision.transforms.functional import to_pil_image
from PIL import ImageDraw

scale_factor = 16.0
learn_rate = 0.001
train_data_path = "dataset/train"
eval_data_path = "dataset/val"
anchor_widths = [8, 32, 64, 128, 256]
aspect_ratios = [0.5, 1.0, 2.0]

img_size = 224

batch_size = 8
num_workers = 0

num_rows = int(img_size / scale_factor)
num_cols = int(img_size / scale_factor)

anchor_grid = get_anchor_grid(
    num_rows,
    num_cols,
    scale_factor,
    anchor_widths,
    aspect_ratios
)

current_batch = 0

def main():
    epochs = input("Please specify the amount of epochs to train:\n")
    while not epochs.isnumeric() or int(epochs) <= 0:
        epochs = input("Epochs must be a whole number greater than 0. Please specify the amount of epochs to train:\n")
    epochs = int(epochs)

    print(f"Beginning training for {epochs} epochs")

    model = SimpleNet(len(anchor_widths), len(aspect_ratios), num_rows, num_cols)
    eval_loader = get_dataloader(train_data_path, img_size, 1, num_workers, anchor_grid, min_iou=0.5, is_test=True)

    global current_batch

    for i, batch in enumerate(eval_loader):
        current_batch = i
        images, _, _ = batch
        get_predictions(model, images, "pre_train")
        print(i)
        if i > 3:
            break

    training_loader = get_dataloader(train_data_path, img_size, 32, num_workers, anchor_grid, min_iou=0.5, is_test=False)
    train_model(model, training_loader, learn_rate, epochs, save=True)

    for i, batch in enumerate(eval_loader):
        current_batch = i
        images, _, _ = batch
        get_predictions(model, images, "post_train")
        if i > 3:
            break

def get_predictions(model, input, dir):

    global current_batch

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    model = model.to(device)
    input = input.to(device)
    model.eval()

    with torch.no_grad():
        prediction, _ = model(input)
        prediction = torch.softmax(prediction, dim=1) # Apply SoftMax for values between 0.0 and 1.0

        batch_size, _, _, _, _, _ = prediction.shape

        human_channel = 1
        human_predictions = prediction[:, human_channel]

        for i in range(batch_size):
            pred_label_grid = human_predictions[i]
            pred_rects = get_pred_rects(anchor_grid, pred_label_grid, 0.5)
            
            img_tensor = input[i]
            img = tensor_to_img(img_tensor)
            draw = ImageDraw.Draw(img)

            for rect in pred_rects:
                
                annotation.draw_annotation(draw, rect.__array__(), color = "red")
            
            img.save(f"test/{dir}/img_b{current_batch}.jpg")

def tensor_to_img(img_tensor: Tensor):
    # Move image back to CPU
    img_tensor = img_tensor.detach().cpu()

    # Denormalize image tensor
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    denorm_tensor = img_tensor * std + mean

    # Convert to PIL Image
    img = to_pil_image(denorm_tensor)
    return img

if __name__ == "__main__":
    main()