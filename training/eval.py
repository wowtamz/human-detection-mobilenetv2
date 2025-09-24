import torch
import numpy as np
import matplotlib.pyplot as plt

from models.simplenet import SimpleNet
from typing import List, Tuple
from datetime import datetime
from itertools import repeat
from utils.annotation import AnnotationRect
from utils.bbr import apply_bbr
from utils.nms import non_maximum_suppression
from assignments.a6.evallib import calculate_ap_pr

# Only import tensorboard if it is installed
try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None

nms_threshold = 0.3
curr_eval_batch = 0
curr_eval_epoch = 0

def get_average_precision(model, loader, device, tag):

    global curr_eval_batch, curr_eval_epoch
    curr_eval_batch, curr_eval_epoch = 0, 0

    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M")

    tensorboard_writer = get_tensorboard_writer(f"avg_precision_{tag}_{timestamp}")

    anchor_grid = loader.dataset.anchor_grid
    try:
        ap = evaluate(model, loader, device, tensorboard_writer, anchor_grid)
    finally:
        if tensorboard_writer:
            tensorboard_writer.close()
    return ap

def batch_inference(
    model: SimpleNet, images: torch.Tensor, device: torch.device, anchor_grid: np.ndarray
) -> List[List[Tuple[AnnotationRect, float]]]:
    
    global nms_threshold, curr_eval_epoch, curr_eval_batch
    
    result = []
    images = images.to(device)
    
    with torch.no_grad():  # Disable gradients for inference speedup
        logits, bbr = model(images)
        prediction = torch.softmax(logits, dim=1)
    
    batch_size, _, _, _, _, _ = prediction.shape
    
    # Pre-compute constants outside the loop
    anchor_flat = anchor_grid.reshape(-1, 4)
    
    # Vectorized score extraction for entire batch
    human_channel = 1
    scores_batch = prediction[:, human_channel].reshape(batch_size, -1)

    # Vectorized bbr adjustments extraction for entire batch
    bbr_batch = bbr.reshape(batch_size, -1, 4) if hasattr(model, "use_bbr") and model.use_bbr else None

    #quit()
    
    for i in range(batch_size):
        print(f"batch inference e:{curr_eval_epoch}/b:{curr_eval_batch}/img:{i}")
        
        # Get scores for current batch item
        scores_flat = scores_batch[i]

        # TODO: Check if bbr_flat is correctly implemented
        # Get bounding box regression adjustments for current batch item
        bbr_flat = bbr_batch[i] if hasattr(model, "use_bbr") and model.use_bbr else None
        
        # Convert scores to numpy for faster list operations
        scores_np = scores_flat.cpu().numpy() if scores_flat.is_cuda else scores_flat.numpy()
        
        if hasattr(model, "use_bbr") and model.use_bbr:
            boxes_scores = [
                (apply_bbr(anchor_box, bbr_adj), score)
                for anchor_box, score, bbr_adj in zip(anchor_flat, scores_np, bbr_flat)
            ]
            score_threshold = 0.05 
            boxes_scores = [bs for bs in boxes_scores if bs[1] > score_threshold] # Filter scores to speed up nms
        else:
            boxes_scores = [
                (AnnotationRect.fromarray(anchor_box), score)
                for anchor_box, score in zip(anchor_flat, scores_np)
            ]
        
        # Apply NMS
        filtered = non_maximum_suppression(boxes_scores, nms_threshold)
        result.append(filtered)
    
    return result

def evaluate(model, loader, device, tensorboard_writer, anchor_grid) -> float:  # feel free to change the arguments
    
    """Evaluates a specified model on the whole validation dataset.
    @return: AP for the validation set as a float.
    """

    global curr_eval_batch
    
    model = model.to(device)
    model.eval()

    dboxes = {}
    gboxes = {}

    gt_dict = loader.dataset.get_annotation_dict()

    with torch.no_grad():

        for i, data in enumerate(loader):

            curr_eval_batch = i
            images, _, ids = data
            
            inference = batch_inference(model, images, device, anchor_grid)

            for j, img_id in enumerate(ids):
                predicted_rects = [(loader.dataset.get_rescaled_annotation(img_id, rect), score) for rect, score in inference[j]]  # Rescale prediction rect to original size
                dboxes[img_id] = predicted_rects
                gboxes[img_id] = gt_dict[img_id]
    
    ap, precision, recall = calculate_ap_pr(dboxes, gboxes)

    if tensorboard_writer:

        plt.figure()
        plt.plot(recall, precision, marker=".")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-Recall Curve")

        tensorboard_writer.add_figure("PR Curve", plt.gcf())
        plt.close()
    
    return ap

def get_tensorboard_writer(model_name):
    if SummaryWriter is not None:
        current_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        tensorboard_writer = SummaryWriter(log_dir=f"tensorboard_logs/{model_name}_{current_time}")
        return tensorboard_writer
    else:
        print("Tensorboard SummaryWriter is not available!")