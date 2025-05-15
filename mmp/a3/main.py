from . import dataset
from mmp.a2 import main as a2
import torch
from pathlib import Path

def main():
    """Put your code for Exercise 3.3 in here"""
    model = a2.MmpNet(len(a2.CLASSES))

    epochs = 5
    img_size = 256
    batch_size = 32
    num_workers = 0

    train_data_dir = f"{Path.cwd()}/dataset/train"
    train_data_loader = dataset.get_dataloader(train_data_dir, img_size, batch_size, num_workers, True)

    loss_func, optimizer = a2.get_criterion_optimizer(model)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    for i in range(epochs):
        print(f"//-- Training epoc {i} --//")
        a2.train_epoch(model, train_data_loader, loss_func, optimizer, device)

    test_data_dir = f"{Path.cwd()}/dataset/test"
    test_data_loader = dataset.get_dataloader(test_data_dir, img_size, batch_size, num_workers, False)
    print(f"\n\n//-- Test Data Benchmarks --//")
    a2.eval_epoch(model, test_data_loader, device)

    
    val_data_dir = f"{Path.cwd()}/dataset/val"
    val_data_loader = dataset.get_dataloader(val_data_dir, img_size, batch_size, num_workers, False)
    print(f"\n\n//-- Validation Data Benchmarks --//")
    a2.eval_epoch(model, val_data_loader, device)

if __name__ == "__main__":
    main()
