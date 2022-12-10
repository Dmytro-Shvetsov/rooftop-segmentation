import torch
import numpy as np
from typing import Any
from segmentation_models_pytorch.losses.tversky import soft_tversky_score


class TverskyScore(torch.nn.Module):
    """Implementation of Tversky score for image segmentation task. 
    Where TP and FP is weighted by alpha and beta params.
    With alpha == beta == 0.5, this score becomes equal DiceScore.

    Args:
        alpha: Weight constant that penalize model for FPs (False Positives)
        beta: Weight constant that penalize model for FNs (False Negatives)
        gamma: Constant that squares the error function. Defaults to ``1.0``
    """

    def __init__(
        self,
        alpha: float = 0.5,
        beta: float = 0.5,
        gamma: float = 1.0,
    ):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def compute_score(self, output, target, smooth=1e-15, eps=1e-7, dims=None) -> torch.Tensor:
        return soft_tversky_score(output, target, self.alpha, self.beta, smooth, eps, dims)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.compute_score(*args, **kwargs)


class IoU(torch.nn.Module):
    def __init__(self, eps=1e-15):
        super().__init__()
        self.eps = eps

    def __call__(self, output, target):
        intersection = (target * output).sum()
        union = target.sum() + output.sum() - intersection
        result = (intersection + self.eps) / (union + self.eps)
        return result


class Precision(torch.nn.Module):
    def __init__(self, eps=1e-15):
        super().__init__()
        self.eps = eps

    def __call__(self, output, target):
        tp = (output * target).sum()
        fp = (output * (1. - target)).sum()
        result = (tp + self.eps) / (tp + fp + self.eps)
        return result


class InstanceAveragePrecision(torch.nn.Module):
    @staticmethod
    def precision_at(threshold, iou):
        """
        Computes the precision at a given threshold.

        Args:
            threshold (float): Threshold.
            iou (np array [n_truths x n_preds]): IoU matrix.

        Returns:
            int: Number of true positives,
            int: Number of false positives,
            int: Number of false negatives.
        """
        matches = iou > threshold
        true_positives = np.sum(matches, axis=1) >= 1  # Correct objects
        false_negatives = np.sum(matches, axis=1) == 0  # Missed objects
        false_positives = np.sum(matches, axis=0) == 0  # Extra objects
        tp, fp, fn = (
            np.sum(true_positives),
            np.sum(false_positives),
            np.sum(false_negatives),
        )
        return tp, fp, fn

    @staticmethod
    def compute_iou(labels, y_pred):
        """
        Computes the IoU for instance labels and predictions.

        Args:
            labels (np array): Labels.
            y_pred (np array): predictions

        Returns:
            np array: IoU matrix, of size true_objects x pred_objects.
        """

        true_objects = len(np.unique(labels))
        pred_objects = len(np.unique(y_pred))

        # Compute intersection areas between all objects
        intersection, xedges, yedges = np.histogram2d(
            labels.flatten(), y_pred.flatten(), bins=(true_objects, pred_objects)
        )

        # Compute areas for true instances (needed for finding the union between all objects)
        area_true, _ = np.histogram(labels, bins=true_objects)

        # Compute areas for predicted instances
        area_pred, _ = np.histogram(y_pred, bins=pred_objects)
        
        area_true = np.expand_dims(area_true, -1)
        area_pred = np.expand_dims(area_pred, 0)

        # Compute union
        union = area_true + area_pred - intersection
        intersection = intersection[1:, 1:]  # exclude background
        union = union[1:, 1:]
        union[union == 0] = 1e-9
        iou = intersection / union

        return iou

    def __call__(self, preds, truths, verbose=0, level=None):
        ious = [self.compute_iou(truth.astype(int), pred.astype(int)) for truth, pred in zip(truths, preds)]

        if verbose:
            print("Thresh\tTP\tFP\tFN\tPrec.")

        if level is not None:
            tps, fps, fns = 0, 0, 0
            for iou in ious:
                tp, fp, fn = self.precision_at(level, iou)
                tps += tp
                fps += fp
                fns += fn
            return tps / (tps + fps + fns)

        prec = []
        for t in np.arange(0.5, 1.0, 0.05):
            tps, fps, fns = 0, 0, 0
            for iou in ious:
                tp, fp, fn = self.precision_at(t, iou)
                tps += tp
                fps += fp
                fns += fn

            p = tps / (tps + fps + fns)
            prec.append(p)

            if verbose:
                print("{:1.3f}\t{}\t{}\t{}\t{:1.3f}".format(t, tps, fps, fns, p))
        
        if verbose:
            print("AP\t-\t-\t-\t{:1.3f}".format(np.mean(prec)))

        return np.mean(prec)
