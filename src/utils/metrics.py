"""Multi-label classification metrics for GO term prediction.

This module provides metric functions for evaluating multi-label predictions,
including accuracy and F1-score calculations for Gene Ontology term assignments.
"""

import torch

def multilabel_accuracy(preds, targets, threshold=0.5):
    """Calculate multi-label classification accuracy.
    
    Args:
        preds: Model predictions (logits)
        targets: Ground truth labels
        threshold: Classification threshold (default: 0.5)
        
    Returns:
        float: Mean accuracy across all labels
    """
    preds = (torch.sigmoid(preds) > threshold).float()
    correct = (preds == targets).float()
    return correct.mean().item()

def multilabel_f1(preds, targets, threshold=0.5):
    """Calculate multi-label F1 score.
    
    Args:
        preds: Model predictions (logits)
        targets: Ground truth labels
        threshold: Classification threshold (default: 0.5)
        
    Returns:
        float: Mean F1 score across all labels
    """
    preds = (torch.sigmoid(preds) > threshold).float()
    tp = (preds * targets).sum(dim=0).float()
    fp = (preds * (1-targets)).sum(dim=0).float()
    fn = ((1-preds) * targets).sum(dim=0).float()
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    return f1.mean().item()
