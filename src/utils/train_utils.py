import os
import torch
import logging
import glob
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def save_checkpoint(
    config, model, optimizer, scheduler, epoch, subontology, best_val_aupr=None
):
    """Save a training checkpoint.

    Args:
        config: Configuration dictionary
        model: ProteinGNN model instance
        optimizer: Optimizer instance
        scheduler: Learning rate scheduler instance
        epoch: Current epoch number
        subontology: Current GO subontology being trained
        best_val_aupr: Best validation AUPR achieved (optional)
    """
    checkpoint_dir = os.path.join(config["run"]["results_dir"], "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "subontology": subontology,
        "config": config,
    }

    if best_val_aupr is not None:
        checkpoint["best_val_aupr"] = best_val_aupr

    # Save epoch-specific checkpoint first if configured (safer for recovery)
    epoch_path = os.path.join(
        checkpoint_dir, f"checkpoint_epoch_{epoch}_{subontology}.pth"
    )
    torch.save(checkpoint, epoch_path)
    logger.info(f"Saved epoch checkpoint to {epoch_path}")

    # Save latest checkpoint
    latest_path = os.path.join(checkpoint_dir, f"checkpoint_latest_{subontology}.pth")
    torch.save(checkpoint, latest_path)
    logger.info(f"Saved checkpoint to {latest_path}")


def load_checkpoint(checkpoint_path, model, optimizer, scheduler, device):
    """Load a training checkpoint.

    Args:
        checkpoint_path: Path to the checkpoint file
        model: ProteinGNN model instance
        optimizer: Optimizer instance
        scheduler: Learning rate scheduler instance
        device: torch.device for computation

    Returns:
        tuple: (start_epoch, config, best_val_aupr)
    """
    logger.info(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Check for newer checkpoints in the same directory
    # This handles cases where checkpoint_latest is stale compared to epoch checkpoints
    if "subontology" in checkpoint:
        subontology = checkpoint["subontology"]
        checkpoint_dir = os.path.dirname(checkpoint_path)

        # Find all epoch checkpoints for this subontology
        pattern = os.path.join(checkpoint_dir, f"checkpoint_epoch_*_{subontology}.pth")
        epoch_files = glob.glob(pattern)

        current_epoch = checkpoint["epoch"]
        newer_checkpoint_path = None

        for path in epoch_files:
            filename = os.path.basename(path)
            # Extract epoch number from filename
            match = re.search(
                rf"checkpoint_epoch_(\d+)_{re.escape(subontology)}\.pth", filename
            )
            if match:
                epoch_num = int(match.group(1))
                if epoch_num > current_epoch:
                    current_epoch = epoch_num
                    newer_checkpoint_path = path

        if newer_checkpoint_path:
            logger.info(
                f"Found newer checkpoint: {newer_checkpoint_path} (epoch {current_epoch}). Reloading."
            )
            checkpoint = torch.load(newer_checkpoint_path, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    start_epoch = checkpoint["epoch"] + 1
    config = checkpoint.get("config")
    best_val_aupr = checkpoint.get("best_val_aupr", 0.0)

    logger.info(f"Resuming from epoch {start_epoch}")
    return start_epoch, config, best_val_aupr
