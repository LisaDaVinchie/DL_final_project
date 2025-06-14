import torch as th
import torch.nn as nn

def select_loss_function(loss_type: str) -> nn.Module:
    """Select the loss function based on the provided type.

    Args:
        loss_type (str): Type of loss function to use. Options are 'mse' for Mean Squared Error and 'l1' for L1 loss.

    Returns:
        nn.Module: The selected loss function module.
    
    Raises:
        ValueError: If the loss_type is not recognized.
    """
    
    if loss_type == 'mse':
        return PerPixelMSE()
    elif loss_type == 'l1':
        return PerPixelL1()
    else:
        raise ValueError(f"Unknown loss type: {loss_type}. Use 'mse' or 'l1'.")
        
class PerPixelMSE(nn.Module):
    def __init__(self):
        """Initialize the Per Pixel MSE loss module."""
        super(PerPixelMSE, self).__init__()
    
    def forward(self, prediction: th.Tensor, target: th.Tensor, masks: th.Tensor, norm: bool = False) -> th.Tensor:
        """Calculate the per-pixel loss between the prediction and the target on masked pixels.

        Args:
            prediction (th.Tensor): output of the model, shape (batch_size, channels, height, width)
            target (th.Tensor): ground truth, shape (batch_size, channels, height, width)
            masks (th.Tensor): binary mask with 0 where the MSE must be calculated, shape (batch_size, channels, height, width).

        Returns:
            th.Tensor: per-pixel MSE loss, non normalized
        """
        
        # Calculate the squared difference, for each pixel
        diff = (prediction - target) ** 2
        
        # Set the masked pixels to 0 where the mask is 1, i.e. where the pixel is not masked
        masked_diff = diff.masked_fill(masks, 0.0)
        sum = masked_diff.sum()
        return sum if not norm else sum / calculate_valid_pixels(masks)  # Return the sum of the squared differences over the number of valid pixels

class PerPixelL1(nn.Module):
    def __init__(self):
        """Initialize the Per Pixel L1 loss module."""
        super(PerPixelL1, self).__init__()
    
    def forward(self, prediction: th.Tensor, target: th.Tensor, masks: th.Tensor) -> th.Tensor:
        """Calculate the per-pixel loss between the prediction and the target, ignoring masked pixels.

        Args:
            prediction (th.Tensor): output of the model, shape (batch_size, channels, height, width)
            target (th.Tensor): ground truth, shape (batch_size, channels, height, width)
            masks (th.Tensor): th.bool mask with False for masked pixels, shape (batch_size, channels, height, width)

        Returns:
            th.Tensor: per-pixel loss
        """
        
        diff = th.abs(prediction - target)
        masked_diff = diff.masked_fill(masks, 0.0)
        return masked_diff.sum()  # Return the sum of the absolute differences over the number of valid pixels
    
def calculate_valid_pixels(masks: th.Tensor) -> int:
    """Calculate the number of valid pixels (not masked) in the masks.

    Args:
        masks (th.Tensor): Boolean mask with False where the pixel is masked, shape (batch_size, channels, height, width).

    Returns:
        int: Number of valid pixels.
    """
    return (~masks).float().sum()
    
def dice_coef(prediction: th.Tensor, target: th.Tensor, mask: th.Tensor):
    inv_mask = (~mask).float()  # Invert the mask to get the pixels that are not masked
    target = (target * inv_mask).float()  # Apply the mask to the target
    prediction = (prediction * inv_mask).float()  # Apply the mask to the prediction
    
    intersection = th.sum(target * prediction)
    union = th.sum(target) + th.sum(prediction)
    
    return intersection, union