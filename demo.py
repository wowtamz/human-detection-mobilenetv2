import sys
import torch

from PIL import Image
from models.simplenet import SimpleNet, BBRNet
from utils.annotation import AnnotationRect, draw_annotation
from utils.anchor_grid import get_anchor_grid
from utils.label_grid import get_pred_rects_scores, get_matching_rects
from utils.dataset import DataLoader, get_dataloader
from utils.transforms import img_to_tensor, tensor_to_img
from utils.nms import non_maximum_suppression
from training.train import train_model
from training.eval import get_average_precision
from PIL import ImageDraw

scale_factor = 16.0
learn_rate = 1e-3
train_data_path = "example_dataset/train"
eval_data_path = "example_dataset/val"
test_data_path = "example_dataset/val"
overfit_path = "example_dataset/overfit"
anchor_widths = [8, 32, 64, 128, 256]
aspect_ratios = [0.5, 1.0, 2.0]

img_size = 224

batch_size = 1
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

test_img_paths = [
    "example_dataset/test/20230823_194816.jpg",
    "example_dataset/val/pexels-bestbe-models-975242-2170387.jpg",
    "example_dataset/val/pexels-pixabay-67674.jpg",
    "example_dataset/val/pexels-pixabay-459976.jpg",
    "example_dataset/val/pexels-amina-165508359-28128693.jpg",
    "example_dataset/val/pexels-vidalbalielojrfotografia-1250643.jpg"
]

def main():

    training_loader = get_dataloader(train_data_path, img_size, batch_size, num_workers, anchor_grid, min_iou=0.5, is_test=False)
    eval_loader = get_dataloader(eval_data_path, img_size, batch_size, num_workers, anchor_grid, min_iou=0.5, is_test=True)
    test_loader = get_dataloader(test_data_path, img_size, batch_size, num_workers, anchor_grid, min_iou=0.5, is_test=True)

    draw_ground_truth_labels(training_loader, "output/ground_truth/")

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = SimpleNet(len(anchor_widths), len(aspect_ratios), num_rows, num_cols)

    # Save predictions of the model prior to training and compute AP
    predict_images(model, test_img_paths, out_path="output/", tag="no_train")
    ap_no_train = get_average_precision(model, training_loader, device, "simple")

    # Overfit the model to one image and save its prediction after training
    overfit_model(model, "example_dataset/test")
    predict_images(model, ["example_dataset/test/20230823_194816.jpg"], out_path="output/", tag="overfit")
    
    # Reinitialize model weights
    model = SimpleNet(len(anchor_widths), len(aspect_ratios), num_rows, num_cols)

    # Train the simple model for specified epochs and save its predictions after training
    train_model(model, training_loader, learn_rate, 25, save=True, use_nm=True)
    predict_images(model, test_img_paths, out_path="output/", tag=f"simple_e25")

    ap_simple_train = get_average_precision(model, training_loader, device, f"simple_train")
    ap_simple_val = get_average_precision(model, eval_loader, device, f"simple_val")
    ap_simple_test = get_average_precision(model, test_loader, device, f"simple_test")

    # Switch to model which uses bounding box regression
    model = BBRNet(len(anchor_widths), len(aspect_ratios), num_rows, num_cols)

    # Train the BBR model for specified epochs and save its predictions after training
    train_model(model, training_loader, learn_rate, 25, save=True, use_nm=True)
    predict_images(model, test_img_paths, out_path="output/", tag=f"bbr_e25")
    ap_bbr_train = get_average_precision(model, training_loader, device, f"bbr_train")
    ap_bbr_val = get_average_precision(model, eval_loader, device, f"bbr_val")
    ap_bbr_test = get_average_precision(model, test_loader, device, f"bbr_test")

    print("AP simple training set prior to training: ", ap_no_train)
    print("AP simple training set: ", ap_simple_train)
    print("AP simple validation set: ", ap_simple_val)
    print("AP simple test set: ", ap_simple_test)
    print("AP bbr training set: ", ap_bbr_train)
    print("AP bbr validation set: ", ap_bbr_val)
    print("AP bbr test set: ", ap_bbr_test)
    print(f"The predictions have been saved inside the '{sys.path[0]}/output' directory.")
    
def draw_ground_truth_labels(loader: DataLoader, dir="output/ground_truth/"):

    for k, batch in enumerate(loader):
        images, labels, ids = batch
        batch_size = labels.shape[0]

        for i in range(batch_size):

            img_tensor = images[i]
            label_grid = labels[i]

            gt_rects = get_matching_rects(anchor_grid, label_grid) if label_grid != None else []
            img = tensor_to_img(img_tensor)
            
            save_prediction(img, [], gt_rects, dir, f"img{k+i}")

    
def overfit_model(model, dataset_path):
    overfit_loader = get_dataloader(dataset_path, img_size, 1, num_workers, anchor_grid, min_iou=0.5, is_test=False)
    train_model(model, overfit_loader, learn_rate, 200, save=True, use_nm=True)

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

def get_predictions(model, input, labels=None, dir="", tag="") -> list[AnnotationRect]:

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