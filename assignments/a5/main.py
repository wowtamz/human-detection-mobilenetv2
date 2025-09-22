import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from datetime import datetime
from .model import MmpNet
from ..a3.annotation import AnnotationRect
from ..a4 import dataset
from ..a4.label_grid import iou
from ..a4 import anchor_grid
from ..a8.bbr import get_bbr_loss

# Only import tensorboard if it is installed
try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None

curr_epoch = 0

def get_preprocessed_bbr_data(labels, anchor_grid, bbr_pred, groundtruths):

    bbr_pred = bbr_pred.cpu()
    labels = labels.cpu()
    batch_size = labels.shape[0]
    anchor_boxes = np.stack([anchor_grid] * batch_size)
    #anchor_boxes_masked = [ab[mask == True] for ab, mask in zip(anchor_boxes, labels)]
    #bbr_pred_masked = [bbr[mask == True] for bbr, mask in zip(bbr_pred, labels)] #bbr_pred[labels == True]
    #anchor_boxes_flat = anchor_boxes[labels == True]

    matched_gts = []
    matched_anchor_boxes = []
    matched_adjustments = []

    # This will keep the batch dimension (result is a list of tensors with varying shapes)

    for i in range(batch_size):
        gts = groundtruths[i]
        #aboxes = anchor_boxes_masked[i]
        #bbr_adjs = bbr_pred_masked[i]

        aboxes = anchor_boxes[i][labels[i] == True]
        bbr_adjs = bbr_pred[i][labels[i] == True]

        for gt in gts:
            max_iou = 0.0
            index = -1
            for j in range(aboxes.shape[0]):
                gt_rect = AnnotationRect.fromarray(gt)
                anchor_rect = AnnotationRect.fromarray(aboxes[j])
                _iou = iou(gt_rect, anchor_rect)
                if _iou > max_iou:
                    max_iou = _iou
                    index = j
            if index >= 0:
                matched_gts.append(torch.tensor(gt.__array__()))
                matched_anchor_boxes.append(torch.tensor(aboxes[index]))
                matched_adjustments.append(bbr_adjs[index])

    return torch.stack(matched_gts), torch.stack(matched_anchor_boxes), torch.stack(matched_adjustments)

def step(
    model: MmpNet,
    criterion,
    optimizer: optim.Optimizer,
    img_batch: torch.Tensor,
    lbl_batch: torch.Tensor,
    neg_mining: bool = False,
    anchor_grid = None,
    groundtruths = None
) -> float:
    """Performs one update step for the model

    @return: The loss for the specified batch. Return a float and not a PyTorch tensor
    """

    optimizer.zero_grad()

    prediction, bbr_pred = model(img_batch)
    loss = criterion(prediction, lbl_batch.long())

    batch_size, _, _, _, _, _ = prediction.shape
        
    # Begin - Exercise 5.3
    if neg_mining:
        mask = get_random_sampling_mask(lbl_batch, 0.1)
        filtered_loss = loss * mask
        loss = filtered_loss.sum() / (mask.sum() + 1e-8) # Prevent division by zero
    else:
        loss = loss.mean()
    # End Ex. 5.3

    if model.use_bbr and bbr_pred != None and anchor_grid.any() and groundtruths != None:
        anchor_boxes_flat, bbr_pred_flat, groundtruths_flat = get_preprocessed_bbr_data(lbl_batch, anchor_grid, bbr_pred, groundtruths)
        bbr_loss = get_bbr_loss(anchor_boxes_flat, bbr_pred_flat, groundtruths_flat)
        loss = loss * bbr_loss

    loss.backward()
    optimizer.step()

    return loss.item() # loss is a single element Tensor

def get_tensorboard_writer(model_name):
    if SummaryWriter is not None:
        current_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        tensorboard_writer = SummaryWriter(log_dir=f"tensorboard_logs/{model_name}_{current_time}")
        return tensorboard_writer
    else:

        print("Tensorboard SummaryWriter is not available!")

def get_random_sampling_mask(labels: torch.Tensor, neg_ratio: float) -> torch.Tensor:
    """
    @param labels: The label tensor that is returned by your data loader.
    The values are either 0 (negative label) or 1 (positive label).
    @param neg_ratio: The desired negative/positive ratio.
    Hint: after computing the mask, check if the neg_ratio is fulfilled.
    @return: A tensor with the same shape as labels
    """
    assert labels.min() >= 0 and labels.max() <= 1  # remove this line if you want
    
    flat_labels = labels.flatten()
    
    pos_mask = flat_labels.bool()
    
    neg_mask = ~pos_mask # Invert positive mask to create negative mask
    neg_indices = torch.nonzero(neg_mask, as_tuple=False).squeeze() # Get indices of all negative samples

    num_pos = pos_mask.sum().item() # Count the number of positive samples
    num_neg_to_sample = int(neg_ratio * num_pos) # Computer the amount of negative samples required

    num_neg_to_sample = min(num_neg_to_sample, neg_indices.numel()) # Dont sample more than available negatives

    selected_neg_indices = neg_indices[torch.randperm(len(neg_indices))[:num_neg_to_sample]] # Randomly select a subset of negative indices
    
    mask = torch.zeros_like(flat_labels, dtype=torch.bool) # Create a boolean mask with same shape as flat_labels and all values False
    # Mask only filters positive samples and selected negative indicies
    mask[pos_mask] = True
    mask[selected_neg_indices] = True

    return mask.view_as(labels) # Reshape mask to original shape of labels

def get_criterion_optimizer(model: nn.Module, learn_rate = 0.002, device = None):
    weights = torch.tensor([0.1, 0.9])
    weights = weights.to(device) if device else weights
    error_func = nn.CrossEntropyLoss(reduction="none", weight=weights)

    optimizer = optim.Adam(model.parameters(), lr=learn_rate)
    return (error_func, optimizer)

def train_epoch(model, loader: dataset.DataLoader, criterion, optimizer, device, neg_mining = False):
    
    global curr_epoch
    curr_batch = 0

    model = model.to(device)
    model.train()

    gt_dict = loader.dataset.get_annotation_dict()
    anchor_grid = loader.dataset.anchor_grid

    for batch in loader:

        images, labels, ids = batch

        images = images.to(device)
        labels = labels.to(device)

        groundtruths = None if not model.use_bbr else [[ann.__array__() for ann in gt_dict[id]] for id in ids] # List of AnnotationRects -> list of arrays
        
        loss = step(model, criterion, optimizer, images, labels.float(), neg_mining, anchor_grid, groundtruths)
        print(f"e:{curr_epoch}/b:{curr_batch}/l:{loss}")
        curr_batch += 1
    curr_epoch += 1

def eval_epoch(eval_epoch, model, loader: dataset.DataLoader, device: torch.device, tensorboard_writer = None) -> float:

    model = model.to(device)
    model.eval()
    total = 0
    correct = 0
    total_loss = 0

    criterion = get_criterion_optimizer(model)[0]

    progress = 0

    gt_dict = loader.dataset.get_annotation_dict()
    anchor_grid = loader.dataset.anchor_grid

    with torch.no_grad():
        for i, data in enumerate(loader):
            print(f"validating epoch {eval_epoch} | iter {progress}")
            images, labels, ids = data
            images = images.to(device)
            labels = labels.to(device)
            labels = labels.float()

            prediction, bbr_pred = model(images)
            loss = criterion(prediction, labels.long())

            if model.use_bbr:
                groundtruths = [[ann.__array__() for ann in gt_dict[id]] for id in ids] # List of AnnotationRects -> list of arrays
                tensor_groundtruths = torch.tensor(groundtruths)

                anchor_boxes = get_anchor_boxes_from_prediction(anchor_grid, prediction)
                bbr_loss = get_bbr_loss(anchor_boxes, bbr_pred, tensor_groundtruths)
                loss = loss * bbr_loss
            
            total += images.shape[0]
            total_loss += loss.item() * images.shape[0]
            correct += torch.sum(prediction == labels)
            progress += 1
    
    loss = total_loss / total
    accuracy = correct / total

    print(f"//===Statistics for epoch {eval_epoch}===//\n Loss: {loss:.4f}, Accuracy: {accuracy * 100:.4f}\n//======================================//")
    if tensorboard_writer:
        tensorboard_writer.add_scalar("Loss/Epoch Training", loss, eval_epoch)
        tensorboard_writer.add_scalar("Accuracy/Epoch Training", accuracy, eval_epoch)
    return accuracy

def train_and_evaluate(neg_mining, anchor_widths, aspect_ratios, scale_factor=8.0, epochs=15, learn_rate=0.002, tag=""):
    
    train_data_path = "dataset/train"
    val_data_path = "dataset/val"
    img_size = 224
    batch_size = 32
    num_workers = 4

    #scale_factor = 8.0 # Default 8.0

    num_rows = int(img_size / scale_factor)
    num_cols = int(img_size / scale_factor)

    #anchor_widths = [4, 8, 16, 32, 64, 128] # [2, 4, 8, 16, 32, 64, 128, 256]
    #aspect_ratios = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0] #[0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2, 3]

    agrid = anchor_grid.get_anchor_grid(
        num_rows,
        num_cols, 
        scale_factor,
        anchor_widths,
        aspect_ratios
    )

    model = MmpNet(len(anchor_widths) ,len(aspect_ratios), num_rows, num_cols)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    train_dataloader = dataset.get_dataloader(train_data_path, img_size, batch_size, num_workers, agrid, False)
    
    eval_dataloader = dataset.get_dataloader(train_data_path, img_size, 1, num_workers, agrid, True)
    loss_func, optimizer = get_criterion_optimizer(model, learn_rate)

    prefix = f"a5_sf{scale_factor}_lr{learn_rate}{tag}"

    tensorboard_writer =  get_tensorboard_writer(prefix + "_neg_mining" if neg_mining else prefix)

    try:
        for i in range(epochs):
            train_epoch(model, train_dataloader, loss_func, optimizer, device, neg_mining)
            #eval_epoch(i, model, eval_dataloader, device, tensorboard_writer)
    finally:
        tensorboard_writer.close()

    torch.save(model.state_dict(), f"{prefix}model.pth")

def main():
    """Put your training code for exercises 5.2 and 5.3 here"""
    
    _tag = "_testing"
    _epochs = 15

    sf = 8.0
    lr = 0.02
    neg_mine = True
    train_and_evaluate(
        scale_factor=sf,
        learn_rate=lr,
        anchor_widths=[4, 8, 16, 32, 64, 128, 224],
        aspect_ratios=[0.1, 0.25, 0.5, 1.0, 1.5, 2.0],
        tag=_tag, neg_mining=neg_mine, epochs=_epochs)
    
if __name__ == "__main__":
    main()
