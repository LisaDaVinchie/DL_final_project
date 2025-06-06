import torch as th
from pathlib import Path
import argparse
import torch as th
import random
import math
import json
import time

mask_name = "masks"
square_mask_name = "square"
lines_mask_name = "lines"

def main():
    start_time = time.time()
    parser = argparse.ArgumentParser(description='Train a CNN model on a dataset')
    parser.add_argument('--paths', type=Path, help='Path to the JSON file with data paths')
    parser.add_argument('--params', type=Path, help='Path to the JSON file with model parameters')

    args = parser.parse_args()

    params_file_path = Path(args.params)
    paths_file_path = Path(args.paths)
    if not params_file_path.exists():
        raise FileNotFoundError(f"Parameters file not found: {params_file_path}")
    if not paths_file_path.exists():
        raise FileNotFoundError(f"Paths file not found: {paths_file_path}")
    
    with open(params_file_path, 'r') as f:
        params = json.load(f)
    with open(paths_file_path, 'r') as f:
        paths = json.load(f)
        
    n_masks = int(params[mask_name]["n_masks"])
        
    mask_dataset_path = Path(paths["next_mask_path"])
    
    if not mask_dataset_path.parent.exists():
        raise FileNotFoundError(f"Mask dataset directory does not exist: {mask_dataset_path.parent}")
    
    print(f"Generating {n_masks} masks\n")
    mask_class = initialize_mask_kind(params)
    
    masks = mask_class.mask(n_masks)
    
    print(f"Generated {n_masks} masks in {time.time() - start_time:.2f} seconds.\n")
    
    print(f"Saving masks to {mask_dataset_path}\n")
    th.save(masks, mask_dataset_path)
    
    elapsed_time = time.time() - start_time
    print(f"Mask generation completed in {elapsed_time:.2f} seconds. Masks saved to {mask_dataset_path}")
    

def initialize_mask_kind(params: dict):
    """Initialize the mask kind based on the provided parameters."""
    mask_kind = params[mask_name]["mask_kind"]
    if mask_kind == square_mask_name:
        return SquareMask(params)
    elif mask_kind == lines_mask_name:
        return LinesMask(params)
    else:
        raise ValueError(f"Unknown mask kind: {mask_kind}")

class SquareMask:
    def __init__(self, params: dict = None, image_nrows: int = None, image_ncols: int = None, mask_percentage: float = None):
        """Initialize the SquareMask class.

        Args:
            params_path (dict, optional): dictionary containing the parameters for the mask. Defaults to None.
            image_nrows (int, optional): number of rows in the image to mask. Defaults to None.
            image_ncols (int, optional): number of columns in the image to mask. Defaults to None.
            mask_percentage (float, optional): fraction of pixels to mask, from 0 to 1. Defaults to None.
        """
        
        self.image_nrows = image_nrows
        self.image_ncols = image_ncols
        self.mask_percentage = mask_percentage
        
        self._initialize_parameters(params)
        
        self._check_parameters()

    def _check_parameters(self):
        if self.image_nrows <= 0 or self.image_ncols <= 0:
            raise ValueError("Image dimensions must be positive integers.")
        
        if self.mask_percentage <= 0 or self.mask_percentage >= 1:
            raise ValueError("Mask percentage must be between 0 and 1.")

    def _initialize_parameters(self, params: dict):
        """Initialize parameters from a dict if provided. Priority is given to the parameters passed in the constructor.

        Args:
            params (dict): dictionary containing the parameters for the mask. It should contain:

        Raises:
            ValueError: If any of the required parameters are None.
        """
        if params is not None:
            self.image_nrows = params['dataset']['nrows'] if self.image_nrows is None else self.image_nrows
            self.image_ncols = params['dataset']['ncols'] if self.image_ncols is None else self.image_ncols
            self.mask_percentage = params[mask_name][square_mask_name]['mask_percentage'] if self.mask_percentage is None else self.mask_percentage
        
        if self.image_nrows is None or self.image_ncols is None or self.mask_percentage is None:
            raise ValueError("Missing one of the following required parameters: image_nrows, image_ncols, mask_percentage")
        
    def mask(self, n: int) -> th.Tensor:
        """Create a square mask of n_pixels in the image

        Returns:
            th.Tensor: binary mask of shape (nrows, ncols), th.bool dtype, where False=masked, True=background
        """
        n_pixels = int(self.mask_percentage * self.image_nrows * self.image_ncols)
        square_nrows = int(n_pixels ** 0.5)
        image_mask = th.ones((n, self.image_nrows, self.image_ncols), dtype=th.bool)
        
        # Get a random top-left corner for the square
        row_idxs = [th.randint(0, self.image_ncols - square_nrows, (1,)).item() for _ in range(n)]
        col_idxs = [th.randint(0, self.image_nrows - square_nrows, (1,)).item() for _ in range(n)]
        
        # Set the square area to False (masked)
        for i in range(n):
            image_mask[i, row_idxs[i]:row_idxs[i] + square_nrows, col_idxs[i]:col_idxs[i] + square_nrows] = False
        
        return image_mask

class LinesMask:
    def __init__(self, params: dict = None, image_nrows: int = None, image_ncols: int = None, num_lines: int = None, min_thickness: int = None, max_thickness: int = None):
        """Initialize the LinesMask class.

        Args:
            params_path (dict, optional): dictionary containing the parameters for the mask. Defaults to None.
            image_nrows (int, optional): number of rows in the image to mask. Defaults to None.
            image_ncols (int, optional): number of columns in the image to mask. Defaults to None.
            num_lines (int, optional): number of lines to generate. Defaults to None.
            min_thickness (int, optional): minimum line thickness. Defaults to None.
            max_thickness (int, optional): maximum line thickness. Defaults to None.
        """
        self.image_nrows = image_nrows
        self.image_ncols = image_ncols
        self.num_lines = num_lines
        self.min_thickness = min_thickness
        self.max_thickness = max_thickness
        
        self._initialize_parameters(params)
        
        self._check_parameters()

    def _check_parameters(self):
        if self.image_nrows <= 0 or self.image_ncols <= 0:
            raise ValueError("Image dimensions must be positive integers.")
        
        if self.num_lines <= 0:
            raise ValueError("Number of lines must be a positive integer.")
        
        if self.min_thickness <= 0 or self.max_thickness < 0:
            raise ValueError("Line thickness must be positive integers.")
        
        if self.min_thickness > self.max_thickness:
            raise ValueError("Minimum thickness cannot be greater than maximum thickness.")
        

    def _initialize_parameters(self, params: dict):
        """Initialize parameters from a dict if provided. Priority is given to the parameters passed in the constructor.

        Args:
            params (dict): Dictionary containing parameters for the mask.

        Raises:
            ValueError: If any of the required parameters are None.
        """
        if params is not None:
                
            self.image_nrows = params['dataset']['nrows'] if self.image_nrows is None else self.image_nrows
            self.image_ncols = params['dataset']['ncols'] if self.image_ncols is None else self.image_ncols
            self.num_lines = params[mask_name][lines_mask_name]['num_lines'] if self.num_lines is None else self.num_lines
            self.min_thickness = params[mask_name][lines_mask_name]['min_thickness'] if self.min_thickness is None else self.min_thickness
            self.max_thickness = params[mask_name][lines_mask_name]['max_thickness'] if self.max_thickness is None else self.max_thickness
        
        if self.image_nrows is None or self.image_ncols is None or self.num_lines is None or self.min_thickness is None or self.max_thickness is None:
            raise ValueError("Missing one of the following required parameters: image_nrows, image_ncols, num_lines, min_thickness, max_thickness")
        
    def mask(self, n: int) -> th.Tensor:
        """Create a mask of lines in the image

        Returns:
            th.Tensor: binary mask of shape (nrows, ncols), th.bool dtype, where False=masked, True=background
        """
        # Start with all ones (background)
        mask = th.ones((n, self.image_nrows, self.image_ncols), dtype=th.bool)
        
        for i in range(n):
            
            for _ in range(self.num_lines):
                # Random start and end points
                start_point = (random.randint(0, self.image_nrows - 1), random.randint(0, self.image_ncols - 1))
                end_point = (random.randint(0, self.image_nrows - 1), random.randint(0, self.image_ncols - 1))
                
                # Random thickness
                thickness = random.randint(self.min_thickness, self.max_thickness)
                
                # Generate the line and subtract from mask (lines become False)
                mask[i, :, :] = mask[i, :, :] * (~self._generate_single_line(start_point, end_point, thickness))
            
        return mask

    def _generate_single_line(self, start_point: tuple, end_point: tuple, thickness: int) -> th.Tensor:
        """Helper function to generate a single line (1=line, 0=background).

        Args:
            start_point (tuple): start point of the line, as (row, col)
            end_point (tuple): end point of the line, as (row, col)
            thickness (int): thickness of the line, in pixels

        Returns:
            th.Tensor: binary mask of the line, of shape (nrows, ncols)
        """
        line_mask = th.zeros((self.image_nrows, self.image_ncols), dtype=th.bool)
        
        y1, x1 = start_point
        y2, x2 = end_point
        
        # Vector from start to end
        dx = x2 - x1
        dy = y2 - y1
        
        # Normalize
        length = max(math.sqrt(dx**2 + dy**2), 1e-8)
        dx /= length
        dy /= length
        
        # Generate points along the line
        num_samples = max(int(length * 2), 2)
        t_values = th.linspace(0, 1, num_samples)
        
        for t in t_values:
            x = x1 + t * (x2 - x1)
            y = y1 + t * (y2 - y1)
            
            # Create a grid for the thickness circle
            radius = thickness / 2
            i_values = th.arange(-thickness//2, thickness//2 + 1, dtype=th.float32)
            j_values = th.arange(-thickness//2, thickness//2 + 1, dtype=th.float32)
            
            for i in i_values:
                for j in j_values:
                    if (i**2 + j**2) <= radius**2:
                        yi = int(th.round(y + i).item())
                        xi = int(th.round(x + j).item())
                        if 0 <= yi < self.image_nrows and 0 <= xi < self.image_ncols:
                            line_mask[yi, xi] = True
                            
        return line_mask
    

if __name__ == "__main__":
    main()