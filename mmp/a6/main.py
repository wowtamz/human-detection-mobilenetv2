from typing import List, Tuple
import torch
import numpy as np

from datetime import datetime
from ..a3.annotation import AnnotationRect
from ..a4.anchor_grid import get_anchor_grid
from ..a4.label_grid import get_label_grid
from ..a4 import dataset
from ..a4.dataset import get_dataloader
from ..a5.model import MmpNet
from ..a5.main import get_criterion_optimizer, get_tensorboard_writer, train_epoch
from .nms import non_maximum_suppression
from .evallib import calculate_ap_pr

# Only import tensorboard if it is installed
try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None

curr_epoch = 0

curr_eval_epoch = 0
curr_eval_batch = 0

nms_threshold = 0.3

def batch_inference(
    model: MmpNet, images: torch.Tensor, device: torch.device, anchor_grid: np.ndarray
) -> List[List[Tuple[AnnotationRect, float]]]:
    
    global curr_eval_epoch
    global curr_eval_batch
    global nms_threshold

    result = []
    images = images.to(device)

    prediction = model(images)
    batch_size, channels, widths, ratios, rows, cols = prediction.shape
       
    anchor_flat = anchor_grid.reshape(-1, 4)
    
    for i in range(batch_size):
        print(f"batch inference e:{curr_eval_epoch}/b:{curr_eval_batch}/img:{i}")
        human_channel = 1
        scores_flat = prediction[i, human_channel].reshape(-1)
        t = 0.5
        filtered_indices = torch.nonzero(scores_flat > t, as_tuple=False).squeeze()
        indices = [filtered_indices.item()] if filtered_indices.dim() == 0 else filtered_indices.tolist()

        boxes_scores = []
        
        for j in indices:
            score = scores_flat[j]
            rect_array = anchor_flat[j]
            rect = AnnotationRect.fromarray(rect_array)
            boxes_scores.append((rect, score))
        
        '''
        for pred_arr, anchor_arr in zip(i_flat_pred, i_flat_anchor):
            score = pred_arr.item()
            anchor_rect = AnnotationRect.fromarray(anchor_arr)
            boxes_scores.append((anchor_rect, score))
        '''

        filtered = non_maximum_suppression(boxes_scores, nms_threshold)
        result.append(filtered)

    return result

def evaluate(model, loader, device, tensorboard_writer, anchor_grid) -> float:  # feel free to change the arguments
    """Evaluates a specified model on the whole validation dataset.

    @return: AP for the validation set as a float.

    You decide which arguments this function should receive
    """

    global curr_eval_batch
    
    model = model.to(device)
    model.eval()

    dboxes = {}
    gboxes = {}

    gt_dict = loader.dataset.get_annotation_dict()

    with torch.no_grad():

        for i, data in enumerate(loader):

            if i == 2:
                break

            curr_eval_batch = i
            images, labels, ids = data
            
            inference = batch_inference(model, images, device, anchor_grid)

            for j, img_id in enumerate(ids):
                dboxes[img_id] = inference[j]
                gboxes[img_id] = gt_dict[img_id]
    
    ap, precision, recall = calculate_ap_pr(dboxes, gboxes)

    if tensorboard_writer:
        for i, p in enumerate(precision):
            r = recall[i]
            tensorboard_writer.add_scalar("Precision/Recall", p, r)
    
    return ap

def evaluate_test(model, data_loader, device, anchor_grid, timestamp):  # feel free to change the arguments
    """Generates predictions on the provided test dataset.
    This function saves the predictions to a text file.

    You decide which arguments this function should receive
    """

    lines = []
    
    model = model.to(device)
    model.eval()

    count = 0
    
    for batch in data_loader:
        images, labels, ids = batch

        inference = batch_inference(model, images, device, anchor_grid)

        for i, img_id in enumerate(ids):

            print(f"Evaluating img {count}")
            prediction = inference[i]

            for box_score in prediction:
                rect = box_score[0]
                score = box_score[1]
                lines.append(f"{img_id} {int(rect.x1)} {int(rect.y1)} {int(rect.x2)} {int(rect.y2)} {score}\n")
        
        count += 1
        
    with open(f"mmp/a6/eval_{timestamp}.txt", "w") as f:
        f.writelines(lines)
        f.close()

def main():
    """Put the surrounding training code here. The code will probably look very similar to last assignment"""
    global curr_eval_epoch
    global nms_threshold

    use_negative_mining = True
    epochs = 50
    scale_factor = 8.0
    learn_rate = 0.005
    train_data_path = "new_dataset/train"
    eval_data_path = "new_dataset/val"
    anchor_widths = [4, 8, 16, 32, 64, 128, 224]
    aspect_ratios = [0.1, 0.25, 0.5, 1.0, 1.5, 2.0]
    img_size = 224
    batch_size = 32
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

    model = MmpNet(len(anchor_widths), len(aspect_ratios), num_rows, num_cols)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    train_data_loader = get_dataloader(train_data_path, img_size, batch_size, num_workers, anchor_grid, False)
    eval_data_loader = get_dataloader(eval_data_path, img_size, 1, num_workers, anchor_grid, True)
    
    loss_func, optimizer = get_criterion_optimizer(model, learn_rate)

    tensorboard_writer = get_tensorboard_writer("a6_exercise_3")

    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M")
    
    try:
        for i in range(epochs):
            curr_eval_epoch = i
            train_epoch(model, train_data_loader, loss_func, optimizer, device, use_negative_mining)
            ap_epoch = evaluate(model, train_data_loader, device, None, anchor_grid)
            tensorboard_writer.add_scalar("Precision/Epoch", ap_epoch, i)
            print(f"Precision on epoch {i}: {ap_epoch}")
            if (i+1) % 5 == 0:
                torch.save(model.state_dict(), f"a6_lr{learn_rate}_e{i}_weights_{timestamp}.pth")

        # load previously trained model here
        #state_dict = torch.load("a5_sf8.0_lr0.02_testingmodel.pth", map_location=torch.device(device))
        #model.load_state_dict(state_dict)
        #evaluate_test(model, eval_data_loader, device, anchor_grid, timestamp)

        ap = evaluate(model, eval_data_loader, device, tensorboard_writer, anchor_grid)
        print(f"Average Precision on validation set: {ap}")

    finally:
        tensorboard_writer.close()
    
if __name__ == "__main__":
    main()
