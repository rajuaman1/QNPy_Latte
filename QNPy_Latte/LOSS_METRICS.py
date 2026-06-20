from sklearn.metrics import mean_squared_error, median_absolute_error
import torch
import torch.nn as nn
from torch.utils.data import IterableDataset, DataLoader, Dataset
from torch.distributions.multivariate_normal import MultivariateNormal
import torch.nn.functional as F

class LogProbLoss(nn.Module):
    """Log probability loss function."""
            
    def __init__(self,latent_samples = 1,param = False):
        super(LogProbLoss, self).__init__()
        self.latent_samples = latent_samples
        self.normal_loss = 0
        self.kl_loss = 0
        self.param = param

    def forward(self, dist, target_y,kl_loss = None,transcribe = False):
        """Returns the loss value.

        Args:
          dist: A multivariate Gaussian over the target points.
          target_y: Array of shape BATCH_SIZE x NUM_TARGET that contains the ground truth y values of the target points.

        Returns:
          loss: Mean over log probabilities.
        """
        log_p = dist.log_prob(target_y)
        if self.latent_samples is None:
            if self.param:
                loss = -torch.mean(log_p.mean(dim = 0).mean(-1))
            else:
                loss = -torch.mean(log_p)
        else:
            if kl_loss is None:
                log_p_summed = torch.logsumexp(log_p, 0)
                full_loss = log_p_summed - torch.log(torch.tensor(self.latent_samples, dtype=log_p.dtype, device=log_p.device))
            else:
                kl_loss = torch.clamp(kl_loss, min=0)
                log_p = log_p.mean(dim = 0).mean(-1)
                full_loss = log_p-kl_loss
            loss = -torch.mean(full_loss)
        return loss
    
class MSELoss(nn.Module):
    """MSE loss function."""
            
    def __init__(self):
        super(MSELoss, self).__init__()

    def forward(self, y_true, y_pred, weights):
        """Returns the MSE value.

        Args:
          y_true: Ground truth values.
          y_pred: Predicted values.
          weights: Measurement errors of ground truth values.

        Returns:
          mse: Mean squared error.
        """
        # Do not include in computation graph
        with torch.no_grad(): 
            # Move to cpu
            y_true, y_pred, weights = y_true.cpu(), y_pred.cpu(), weights.cpu()
            mse = mean_squared_error(y_true, y_pred, sample_weight=weights, multioutput='uniform_average')
            return mse

class MAELoss(nn.Module):
    """MAE loss function."""
            
    def __init__(self):
        super(MAELoss, self).__init__()

    def forward(self, y_true, y_pred, weights):
        """Returns the MAE value.

        Args:
          y_true: Ground truth values.
          y_pred: Predicted values.
          weights: Measurement errors of ground truth values.

        Returns:
          mse: Mean squared error.
        """
        # Do not include in computation graph
        with torch.no_grad(): 
            # Move to cpu
            y_true, y_pred, weights = y_true.cpu(), y_pred.cpu(), weights.cpu()
            mae = median_absolute_error(y_true, y_pred,  multioutput='uniform_average')
            return mae