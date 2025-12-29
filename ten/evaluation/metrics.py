"""
Evaluation Metrics
==================

Metrics for classification and regression tasks in drug discovery.
"""

import numpy as np
from typing import Union


def calculate_auroc(
    predictions: np.ndarray,
    labels: np.ndarray,
) -> float:
    """
    Calculate Area Under ROC Curve.
    
    Args:
        predictions: Predicted probabilities
        labels: True binary labels
    
    Returns:
        AUROC score
    """
    try:
        from sklearn.metrics import roc_auc_score
        
        # Handle multi-task case
        if predictions.ndim > 1 and predictions.shape[1] > 1:
            scores = []
            for i in range(predictions.shape[1]):
                valid_mask = ~np.isnan(labels[:, i])
                if valid_mask.sum() > 0:
                    score = roc_auc_score(labels[valid_mask, i], predictions[valid_mask, i])
                    scores.append(score)
            return np.mean(scores) if scores else 0.0
        else:
            predictions = predictions.squeeze()
            labels = labels.squeeze()
            valid_mask = ~np.isnan(labels)
            return roc_auc_score(labels[valid_mask], predictions[valid_mask])
    except Exception:
        return 0.0


def calculate_auprc(
    predictions: np.ndarray,
    labels: np.ndarray,
) -> float:
    """
    Calculate Area Under Precision-Recall Curve.
    
    Args:
        predictions: Predicted probabilities
        labels: True binary labels
    
    Returns:
        AUPRC score
    """
    try:
        from sklearn.metrics import average_precision_score
        
        if predictions.ndim > 1 and predictions.shape[1] > 1:
            scores = []
            for i in range(predictions.shape[1]):
                valid_mask = ~np.isnan(labels[:, i])
                if valid_mask.sum() > 0:
                    score = average_precision_score(labels[valid_mask, i], predictions[valid_mask, i])
                    scores.append(score)
            return np.mean(scores) if scores else 0.0
        else:
            predictions = predictions.squeeze()
            labels = labels.squeeze()
            valid_mask = ~np.isnan(labels)
            return average_precision_score(labels[valid_mask], predictions[valid_mask])
    except Exception:
        return 0.0


def calculate_rmse(
    predictions: np.ndarray,
    labels: np.ndarray,
) -> float:
    """
    Calculate Root Mean Squared Error.
    
    Args:
        predictions: Predicted values
        labels: True values
    
    Returns:
        RMSE
    """
    predictions = predictions.squeeze()
    labels = labels.squeeze()
    
    valid_mask = ~np.isnan(labels) & ~np.isnan(predictions)
    
    mse = np.mean((predictions[valid_mask] - labels[valid_mask]) ** 2)
    return np.sqrt(mse)


def calculate_mae(
    predictions: np.ndarray,
    labels: np.ndarray,
) -> float:
    """
    Calculate Mean Absolute Error.
    
    Args:
        predictions: Predicted values
        labels: True values
    
    Returns:
        MAE
    """
    predictions = predictions.squeeze()
    labels = labels.squeeze()
    
    valid_mask = ~np.isnan(labels) & ~np.isnan(predictions)
    
    return np.mean(np.abs(predictions[valid_mask] - labels[valid_mask]))


def calculate_accuracy(
    predictions: np.ndarray,
    labels: np.ndarray,
    threshold: float = 0.5,
) -> float:
    """
    Calculate classification accuracy.
    
    Args:
        predictions: Predicted probabilities
        labels: True binary labels
        threshold: Decision threshold
    
    Returns:
        Accuracy
    """
    predictions = predictions.squeeze()
    labels = labels.squeeze()
    
    valid_mask = ~np.isnan(labels)
    
    pred_labels = (predictions[valid_mask] >= threshold).astype(float)
    
    return np.mean(pred_labels == labels[valid_mask])


def calculate_f1(
    predictions: np.ndarray,
    labels: np.ndarray,
    threshold: float = 0.5,
) -> float:
    """
    Calculate F1 score.
    
    Args:
        predictions: Predicted probabilities
        labels: True binary labels
        threshold: Decision threshold
    
    Returns:
        F1 score
    """
    try:
        from sklearn.metrics import f1_score
        
        predictions = predictions.squeeze()
        labels = labels.squeeze()
        
        valid_mask = ~np.isnan(labels)
        
        pred_labels = (predictions[valid_mask] >= threshold).astype(int)
        
        return f1_score(labels[valid_mask].astype(int), pred_labels)
    except Exception:
        return 0.0


def calculate_perplexity(loss: float) -> float:
    """
    Calculate perplexity from cross-entropy loss.
    
    Args:
        loss: Cross-entropy loss value
    
    Returns:
        Perplexity
    """
    return np.exp(min(loss, 20))  # Cap for numerical stability


def calculate_convergence_speed(
    losses: list,
    target_loss: float,
) -> int:
    """
    Calculate steps to reach target loss.
    
    Args:
        losses: List of loss values per step
        target_loss: Target loss to reach
    
    Returns:
        Number of steps to reach target, or -1 if not reached
    """
    for i, loss in enumerate(losses):
        if loss <= target_loss:
            return i
    return -1
