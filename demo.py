import torch
import gc

from PIL import Image
from models.simplenet import SimpleNet
from utils.annotation import AnnotationRect, draw_annotation
from utils.anchor_grid import get_anchor_grid
from utils.label_grid import get_pred_rects_scores, get_matching_rects
from utils.dataset import get_dataloader
from utils.transforms import img_to_tensor, tensor_to_img
from utils.nms import non_maximum_suppression
from training.train import train_model
from PIL import ImageDraw

scale_factor = 16.0
learn_rate = 1e-3
train_data_path = "dataset/train"
eval_data_path = "dataset/val"
overfit_path = "overfit"
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

eval_img_paths = [
    "overfit/02241596.jpg",
    "dataset/train/02241588.jpg",
    "dataset/train/02241589.jpg",
    "dataset/train/02241600.jpg",
]

current_batch = 0

def main():
    epochs = input("Please specify the amount of epochs to train:\n")
    while not epochs.isnumeric() or int(epochs) <= 0:
        epochs = input("Epochs must be a whole number greater than 0. Please specify the amount of epochs to train:\n")
    epochs = int(epochs)

    print(f"Beginning training for {epochs} epochs")

    model = SimpleNet(len(anchor_widths), len(aspect_ratios), num_rows, num_cols)

    # Save predictions of the model prior to training
    predict_images(model, eval_img_paths, out_path="test/pre_train/")

    # Overfit the model to one image and save its prediction after training
    overfit_model(model)
    predict_images(model, eval_img_paths, out_path="test/post_train/", tag="overfit_e100")

    # Traing the model for specified epochs and save its predictions after training
    training_loader = get_dataloader(train_data_path, img_size, 32, num_workers, anchor_grid, min_iou=0.5, is_test=False)
    train_model(model, training_loader, learn_rate, epochs, save=True, use_nm=True)
    predict_images(model, eval_img_paths, out_path="test/post_train/", tag=f"nn_e{epochs}")

    # Load a pre-trained model's weights and save its predictions
    load_model_weights(model, "model_e20_09-25.pth")
    predict_images(model, eval_img_paths, out_path="test/post_train/", tag="nn_e20")
    
def overfit_model(model):
    overfit_loader = get_dataloader(overfit_path, img_size, 1, num_workers, anchor_grid, min_iou=0.5, is_test=False)
    train_model(model, overfit_loader, learn_rate, 100, save=True, use_nm=True)

def predict_images(model, img_paths: list, out_path: str, tag=""):
    img_list = [Image.open(path) for path in img_paths]
    img_tensor_list = [img_to_tensor(img, 224) for img in img_list]
    img_batch = torch.stack(img_tensor_list)
    predicted_rects = get_predictions(model, img_batch, dir=out_path, tag=tag)

def load_model_weights(model: torch.nn.Module, path: str):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    try:
        model_state = torch.load(path, map_location=torch.device(device))
        model.load_state_dict(model_state)
    except FileNotFoundError:
        print(f"Warning, failed to load model state at '{path}', continuing with initial weights!")

def save_prediction(img, prediction: list, ground_truth: list = [], dir="test/", name="img"):
    
    draw = ImageDraw.Draw(img)
    for rect in prediction:
        draw_annotation(draw, rect.__array__(), color = "red")
    
    for rect in ground_truth:
        draw_annotation(draw, rect.__array__(), color = "green")
    
    img.save(f"{dir}{name}.jpg")

def get_predictions(model, input, labels=None, dir="", tag=""):

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

        predicted_rects = []

        for i in range(batch_size):
            pred_label_grid = human_predictions[i]
            rects_scores = get_pred_rects_scores(anchor_grid, pred_label_grid, 0.5)

            nms_rects_scores = non_maximum_suppression(rects_scores, 0.2)

            pred_rects = [rect_score[0] for rect_score in nms_rects_scores]
            predicted_rects.append(pred_rects)

            img_tensor = input[i]
            img = tensor_to_img(img_tensor)

            label_grid = labels[i] if labels != None else None
            gt_rects = get_matching_rects(anchor_grid, label_grid) if label_grid != None else []

            save_prediction(img, pred_rects, gt_rects, dir=dir, name=f"img{i}_{tag}")
        
        return predicted_rects

if __name__ == "__main__":
    main()