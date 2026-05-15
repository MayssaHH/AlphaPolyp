# Backward-compatibility shim: predict.py and test.py import from this module.
# The canonical implementation lives in LossFunctions.py.
from model_architecture.LossFunctions import dice_metric_loss, normalized_mse_loss

__all__ = ["dice_metric_loss", "normalized_mse_loss"]
