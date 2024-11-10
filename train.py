import os
from collections import defaultdict

import matplotlib.pyplot as plt
import torch
import torchmetrics
from torch.utils.data import DataLoader
from torchvision.transforms import v2
from tqdm.auto import tqdm

from dataset import INPUT_VARS, WildfireDataset
from model import create_deeplabv3

USE_WANDB = True
WANDB_DIR = os.path.expandvars("$SCRATCH/wandb")
NPZ_DIR = "arneshdadhich/Downloads/converted.npz" #path to the npz file

if USE_WANDB:
    import wandb


def main():
    if USE_WANDB:
        wandb.init(
            project="wildfire_segmentation",
            sync_tensorboard=True,
            save_code=True,
            dir=WANDB_DIR,
        )
    writer = torch.utils.tensorboard.SummaryWriter()
    print(f"Initialized Tensorboard summary writer in: {writer.log_dir}")

    epochs = 300
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_deeplabv3().to(device)
    model = torch.compile(model)  # mode="reduce-overhead" or "max-autotune"
    train_loader, val_loader = create_data_loaders(NPZ_DIR, use_transforms=True)

    # change this to test different algorithms
    loss = torch.nn.BCEWithLogitsLoss(reduction="mean", pos_weight=torch.tensor(100.0))
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    # fmt: off
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=0.01, total_steps=epochs * len(train_loader), pct_start=0.3,
        base_momentum=0.85, max_momentum=0.95, div_factor=25, final_div_factor=1e4,
    )
    
    train_metrics = initialize_metrics().to(device)
    val_metrics = initialize_metrics().to(device)
    
    ### code below written with help of AI (GitHub copilot)
    # Train & validate for N epochs
    for epoch in tqdm(range(epochs), desc="Epochs", total=epochs, leave=False):
        #========= TRAIN SET ===========#
        train_results = run_one_epoch(
            model, train_loader, loss, optimizer,
            device, train_metrics, scheduler, is_train=True
        )
        writer.add_scalar("train/lr", scheduler.get_last_lr()[0], epoch)
        [writer.add_scalar(f"train/{k}", v, epoch) for k, v in train_results.items()]
        
        #========= VALIDATION SET ===========#
        val_results = run_one_epoch(
            model, val_loader, loss, None, device, val_metrics, None, is_train=False
        )
        [writer.add_scalar(f"val/{k}", v, epoch) for k, v in val_results.items()]
        if USE_WANDB:
            log_images(epoch, model, val_loader, device)

        if epoch % 10 == 0:
            torch.save(model.state_dict(), os.path.join(writer.log_dir, f"model_epoch_{epoch}.pth"))
    # fmt: on
    writer.close()
    if USE_WANDB:
        wandb.finish()


def create_data_loaders(npz_dir, use_transforms=False):
    transforms = v2.Compose(
        [
            v2.RandomPhotometricDistort(p=1),
            v2.RandomResizedCrop(size=(64, 64), scale=(0.5, 1.0), ratio=(0.75, 1.333)),
            v2.RandomHorizontalFlip(p=1),
            v2.ToDtype(torch.float32, scale=True),
        ]
    )
    transforms = None if not use_transforms else transforms
    train_dataset = WildfireDataset(
        data_file=os.path.join(npz_dir, "train.npz"),
        transforms=transforms,
    )
    val_dataset = WildfireDataset(os.path.join(npz_dir, "val.npz"))
    loader_kwargs = dict(batch_size=1024, num_workers=8, prefetch_factor=4)
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
    return train_loader, val_loader


def initialize_metrics():
    return torchmetrics.MetricCollection(
        {
            "loss": torchmetrics.MeanMetric(),
            "accuracy": torchmetrics.Accuracy(task="binary"),
            "f1_score": torchmetrics.F1Score(
                task="binary", num_classes=1, threshold=0.5
            ),
            "recall": torchmetrics.Recall(task="binary", num_classes=1, threshold=0.5),
            "precision": torchmetrics.Precision(
                task="binary", num_classes=1, threshold=0.5
            ),
        }
    )


def run_one_epoch(
    model, loader, criterion, optimizer, device, metrics, scheduler, is_train=True
):
    if is_train:  # enable gradient tracking and updates
        model.train()
        context = torch.enable_grad()
    else:  # no gradient updates/tracking during eval
        model.eval()
        context = torch.no_grad()

    with context:
        progress_bar = tqdm(loader, total=len(loader), desc="Batches", leave=False)
        for inputs, targets in progress_bar:
            inputs, targets = inputs.to(device), targets.to(device)
            if is_train:
                optimizer.zero_grad()
            outputs = model(inputs)
            valid_outputs, valid_targets = process_outputs_targets(outputs, targets)
            loss = criterion(valid_outputs, valid_targets)
            if is_train:
                loss.backward()
                optimizer.step()
                scheduler.step()

            progress_bar.set_postfix({"loss": loss.item()})

            # Update metrics
            metrics["loss"].update(loss)
            metrics["accuracy"].update(valid_outputs, valid_targets.long())
            metrics["f1_score"].update(valid_outputs, valid_targets.long())
            metrics["recall"].update(valid_outputs, valid_targets.long())
            metrics["precision"].update(valid_outputs, valid_targets.long())
       

    results = metrics.compute()
    metrics.reset()
    return results

### code below written with help of AI (GitHub copilot)
def process_outputs_targets(outputs, targets):
    outputs = outputs["out"].squeeze(1)
    targets = targets.squeeze(1)
    mask = targets != -1
    if mask.any():
        valid_outputs, valid_targets = outputs[mask], targets[mask]
        return valid_outputs, valid_targets
    else:
        return outputs, targets


def log_images(epoch, model, loader, device, num_images=5, input_vars=INPUT_VARS):
    """
    Logs images and segmentation masks to W&B for visualization and analysis.

    Args:
        epoch (int): The current epoch number
        model: The trained model
        loader: The data loader for loading images and targets
        device: The device to perform computations on
        num_images (int, optional): The maximum number of images to log. Defaults to 5.
        input_vars (list, optional): The list of input variables to log. Defaults to INPUT_VARS.
    """
    model.eval()
    images_logged = 0
    wandb_images = defaultdict(list)
    with torch.no_grad():
        for inputs, targets in loader:
            if images_logged >= num_images:
                break  # exit if we've logged enough images
            inputs, targets = inputs.to(device), targets.to(device)  # shape: (B,C,H,W)
            outputs = model(inputs)["out"].squeeze(1)
            prediction_masks = (outputs.sigmoid() > 0.5).int().cpu().numpy()
            ground_truth_masks = targets.int().cpu().numpy()
            ground_truth_masks[ground_truth_masks == -1] = 2
            # loop across samples in the batch (i.e., along B dimension)
            for i in range(min(num_images - images_logged, inputs.size(0))):
                for idx, var_name in enumerate(input_vars):  # plot each input variable
                    var = inputs[i, idx].cpu().numpy()
                    # Normalize the input channels for better visualization
                    var = (var - var.min()) / (var.max() - var.min())
                    # Apply viridis color map and convert to RGB (dropping alpha channel)
                    colored_image = plt.get_cmap("viridis")(var)[:, :, :3]

                    # Log images and segmentation masks to W&B
                    wandb_image = wandb.Image(
                        colored_image,
                        caption=var_name,
                        masks={
                            "predictions": {
                                "mask_data": prediction_masks[i].squeeze(),
                                "class_labels": {0: "Not Fire", 1: "Fire"},
                            },
                            # fmt: off
                            "ground_truth": {
                                "mask_data": ground_truth_masks[i].squeeze(),
                                "class_labels": {0: "Not Fire", 1: "Fire", 2: "Unlabeled"},
                            },
                            # fmt: on
                        },
                    )
                    wandb_images[f"val/images/{var_name}"].append(wandb_image)
                images_logged += 1
    wandb.log(wandb_images, step=epoch)


if __name__ == "__main__":
    main()