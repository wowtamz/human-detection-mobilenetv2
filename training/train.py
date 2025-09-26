import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gc

from torch.utils.data import DataLoader
from datetime import datetime
from itertools import repeat
from models.simplenet import SimpleNet
from utils import dataset
from utils.annotation import AnnotationRect
from utils.bbr import get_bbr_loss
from utils.label_grid import iou

t_epoch: int = 0

def train_model(model: SimpleNet, training_dataloader: DataLoader, learnrate: float = 0.001, epochs: int = 50, use_nm: bool = True, use_bbr: bool = False, save: bool = False):
    
    bbr_tag = "_bbr" if use_bbr else ""
    tag = f"model{bbr_tag}_e{epochs}"

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    loss_func, optimizer = get_criterion_optimizer(model, learnrate, device)
    
    train(epochs, model, loss_func, optimizer, device, training_dataloader, use_nm, evaluate_training=False, tag=tag)

    if save:
        timestamp = datetime.now().strftime("%Y-%m-%d-%H:%M")
        torch.save(model.state_dict(), f"{tag}_{timestamp}.pth")

    del optimizer
    torch.cuda.empty_cache()
    gc.collect()

def train(epochs, model, loss_func, optimizer, device, loader, negative_mining = True, evaluate_training = False, augments = "", tag=""):

    global t_epoch
    t_epoch = 0

    for _ in repeat(None, epochs):
        train_epoch(model, loader, loss_func, optimizer, device, negative_mining)

def step(
    model: SimpleNet,
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
    
    if neg_mining:
        mask = get_random_sampling_mask(lbl_batch, 5.0)
        
        filtered_loss = loss * mask
        loss = filtered_loss.sum() / (mask.sum() + 1e-8) # Prevent division by zero
    else:
        #loss = focal_loss(prediction, lbl_batch.long())
        loss = loss.mean()

    if hasattr(model, "use_bbr") and model.use_bbr and bbr_pred != None and anchor_grid.any() and groundtruths != None:
        anchor_boxes_flat, bbr_pred_flat, groundtruths_flat = get_preprocessed_bbr_data(lbl_batch, anchor_grid, bbr_pred, groundtruths)
        bbr_loss = get_bbr_loss(anchor_boxes_flat, bbr_pred_flat, groundtruths_flat)
        loss = loss * bbr_loss

    loss.backward()
    optimizer.step()

    return loss.item() # loss is a single element Tensor

def focal_loss(logits, targets, alpha=0.25, gamma=2.0):
    C = logits.size(1)
    target_one_hot = torch.nn.functional.one_hot(targets, num_classes=C)
    target_one_hot = target_one_hot.permute(0, -1, *range(1, targets.ndim)).float()
    bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, target_one_hot, reduction="none")
    pt = torch.exp(-bce)
    return (alpha * (1 - pt) ** gamma * bce).mean()

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

def get_preprocessed_bbr_data(labels, anchor_grid, bbr_pred, groundtruths):

    bbr_pred = bbr_pred.cpu()
    labels = labels.cpu()
    batch_size = labels.shape[0]
    anchor_boxes = np.stack([anchor_grid] * batch_size)

    matched_gts = []
    matched_anchor_boxes = []
    matched_adjustments = []

    # This will keep the batch dimension (result is a list of tensors with varying shapes)

    for i in range(batch_size):
        gts = groundtruths[i]

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


def get_criterion_optimizer(model: nn.Module, learn_rate = 0.002, device = None):
    error_func = nn.CrossEntropyLoss(reduction="none")

    optimizer = optim.Adam(model.parameters(), lr=learn_rate)
    return (error_func, optimizer)

def train_epoch(model, loader: dataset.DataLoader, criterion, optimizer, device, neg_mining = False):
    
    global t_epoch
    curr_batch = 0

    model = model.to(device)
    model.train()

    gt_dict = loader.dataset.get_annotation_dict()
    anchor_grid = loader.dataset.anchor_grid

    for batch in loader:

        images, labels, ids = batch

        images = images.to(device)
        labels = labels.to(device)

        groundtruths = None if not hasattr(model, "use_bbr") or (hasattr(model, "use_bbr") and not model.use_bbr) else [[ann.__array__() for ann in gt_dict[id]] for id in ids] # List of AnnotationRects -> list of arrays
        
        loss = step(model, criterion, optimizer, images, labels.float(), neg_mining, anchor_grid, groundtruths)
        print(f"e:{t_epoch}/b:{curr_batch}/l:{loss}")
        curr_batch += 1
    t_epoch += 1