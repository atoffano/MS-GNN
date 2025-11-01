import os
import torch
import logging

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

    # Save latest checkpoint
    latest_path = os.path.join(checkpoint_dir, f"checkpoint_latest_{subontology}.pth")
    torch.save(checkpoint, latest_path)
    logger.info(f"Saved checkpoint to {latest_path}")

    # Save epoch-specific checkpoint if configured
    if config["run"].get("save_epoch_checkpoints", False):
        epoch_path = os.path.join(
            checkpoint_dir, f"checkpoint_epoch_{epoch}_{subontology}.pth"
        )
        torch.save(checkpoint, epoch_path)
        logger.info(f"Saved epoch checkpoint to {epoch_path}")


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

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    start_epoch = checkpoint["epoch"] + 1
    config = checkpoint.get("config")
    best_val_aupr = checkpoint.get("best_val_aupr", 0.0)

    logger.info(f"Resuming from epoch {start_epoch}")
    return start_epoch, config, best_val_aupr
