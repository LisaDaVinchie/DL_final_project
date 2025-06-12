import torch as th
import torch.optim as optim
from pathlib import Path
import argparse
from time import time
import json
from import_dataset import create_dataloaders, load_cifar10_data
from model import select_model
from losses import select_loss_function, dice_coef, calculate_valid_pixels
import json

def main():
    """Main function to train a model on a dataset."""
    start_time = time()
    
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
        
    params_str = json.dumps(params, indent=4)[1: -1]
    
    weights_path = Path(paths["next_weights_path"])
    results_path = Path(paths["next_result_path"])
    masks_path = Path(paths["current_mask_path"])
    masks_dir = masks_path.parent
    
    if not weights_path.parent.exists():
        raise FileNotFoundError(f"Weights directory does not exist: {weights_path.parent}")
    if not results_path.parent.exists():
        raise FileNotFoundError(f"Results directory does not exist: {results_path.parent}")
    if not masks_dir.exists():
        raise FileNotFoundError(f"Masks dir does not exist: {masks_dir}")
    
    # Ensure the results file exists and is a txt file
    results_path = results_path.with_suffix('.txt')
    results_path.touch(exist_ok=True)
    
    training_params = params["training"]
    model_name = str(training_params["model_name"])
    batch_size = int(training_params["batch_size"])
    loss_kind = str(training_params["loss_kind"])
    epochs = int(training_params["epochs"])
    learning_rate = float(training_params["learning_rate"])
    mask_idx = int(training_params["mask_idx"])
    n_train = int(training_params["n_train"])
    n_test = int(training_params["n_test"])
    classes = list(training_params["classes"])
    if n_train <= 0:
        n_train = None
    if n_test <= 0:
        n_test = None
    if len(classes) == 0:
        classes = None
    lr_scheduler = str(training_params["lr_scheduler"])
    
    masks_path = masks_dir / f"mask_{mask_idx}.pt"
    
    if not masks_path.exists():
        raise FileNotFoundError(f"Masks file {masks_path} does not exist")
        
    
    model = select_model(model_name)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss = select_loss_function(loss_kind)
    
    scheduler = None
    schedulers_params = params["lr_schedulers"]
    if lr_scheduler == "lambda":
        factor = float(schedulers_params[lr_scheduler]["factor"])
        step_size = int(schedulers_params[lr_scheduler]["step_size"])
        lr_lambda = lambda step: factor ** (step // step_size)
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    elif lr_scheduler == "step":
        step_size = int(schedulers_params[lr_scheduler]["step_size"])
        gamma = float(schedulers_params[lr_scheduler]["gamma"])
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    elif lr_scheduler != "none":
        raise ValueError(f"Unknown learning rate scheduler: {lr_scheduler}.")
        
    train = TrainModel(model=model,
                      loss_function=loss,
                      optimizer=optimizer,
                      results_path = results_path,
                      weights_path = weights_path,
                      lr_scheduler=scheduler)
    
    trainset, testset = load_cifar10_data(normalize=True, n_train = n_train, n_test = n_test, desired_classes=[0])
    
    print(f"Training on {len(trainset)} images, testing on {len(testset)} images", flush=True)
    
    masks = th.load(masks_path)
    
    train_loader, test_loader = create_dataloaders(trainset, testset, batch_size=batch_size)
    
    train.train(train_loader, test_loader, masks, epochs)
    
    elapsed_time = time() - start_time
    print(flush=True)
    train.save_weights()
    print(f"Model weights saved at {weights_path}", flush=True)
    
    print(flush=True)
    train.save_results(params_str, elapsed_time)
    print(f"Results saved at {results_path}", flush=True)
    
    print(f"\nTraining completed in {elapsed_time:.2f} seconds", flush=True)
    
class TrainModel:
    def __init__(self, model: th.nn.Module, loss_function: th.nn.Module, optimizer: optim, results_path: Path, weights_path: Path, save_every: int = 1, lr_scheduler: optim.lr_scheduler = None):
        """Initialize the training class.

        Args:
            model (th.nn.Module): the model to train
            loss_function (th.nn.Module): the loss function to use
            optimizer (optim.Optimizer): the optimizer to use
            lr_scheduler (optim.lr_scheduler): learning rate scheduler
            results_path (Path): path to save the training results
            weights_path (Path): path to save the model weights
            save_every (int): how often to save the model weights and results
        """
        
        self.model = model
        self.loss_function = loss_function
        self.device = th.device("cuda" if th.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.optimizer = optimizer
        self.lr = self.optimizer.param_groups[0]['lr']
        self.save_every = save_every
        self.results_path = results_path
        self.weights_path = weights_path
        self.lr_scheduler = lr_scheduler
        
        self.train_losses = []
        self.test_losses = []
        self.training_lr = []
        self.train_dice_coef = []
        self.test_dice_coef = []
        
    def train(self, train_loader, test_loader, masks, epochs: int):
        """Train the model on the dataset.

        Args:
            train_loader (th.utils.data.DataLoader): training dataloader
            test_loader (th.utils.data.DataLoader): testing dataloader
        """
        
        if epochs <= 0:
            raise ValueError("Number of epochs must be positive.")
        
        n_masks = masks.shape[0]
        
        print(flush=True)
        for epoch in range(epochs):
            print(f"Epoch {epoch + 1}/{epochs}\n", flush=True)
            self.model.train()
            epoch_loss = 0.0
            epoch_intersection = 0.0
            epoch_union = 0.0
            n_valid_pixels = 0
            for (images, _) in train_loader:
                
                n_img = images.shape[0]
                
                random_idxs = th.randint(0, n_masks, (n_img,))
                
                batch_masks = masks[random_idxs].unsqueeze(1).repeat(1, 3, 1, 1)
                
                loss, inter, un = self._compute_loss(images, batch_masks)
                epoch_loss += loss.item()
                n_valid_pixels += calculate_valid_pixels(batch_masks).item()
                epoch_intersection += inter.item()
                epoch_union += un.item()
                
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
            
            self.lr_scheduler.step() if self.lr_scheduler is not None else None

            self.training_lr.append(self.optimizer.param_groups[0]['lr'])
            # self.scheduler.step() if self.scheduler is not None else None
            
            self.train_losses.append(epoch_loss / n_valid_pixels)
            self.train_dice_coef.append(2 * (epoch_intersection + 1e-7) / (epoch_union + 1e-7))
            
            with th.no_grad():
                self.model.eval()
                epoch_loss = 0.0
                n_valid_pixels = 0
                epoch_intersection = 0.0
                epoch_union = 0.0
                for (images, _) in test_loader:
                    n_img = images.shape[0]
                    random_idxs = th.randint(0, n_masks, (n_img,))
                    batch_masks = masks[random_idxs].unsqueeze(1).repeat(1, 3, 1, 1)
                    loss, inter, un = self._compute_loss(images, batch_masks)
                    epoch_loss += loss.item()
                    n_valid_pixels += calculate_valid_pixels(batch_masks).item()
                    epoch_intersection += inter.item()
                    epoch_union += un.item()
                    
                self.test_losses.append(epoch_loss / n_valid_pixels)
                self.test_dice_coef.append(2 * (epoch_intersection + 1e-7) / (epoch_union + 1e-7))
            
            if (epoch + 1) % self.save_every == 0:
                self.save_weights()
                self.save_results()
                print(f"\nModel weights and results saved at epoch {epoch + 1}\n", flush=True)
                
        print(flush=True)
                
    def _compute_loss(self, images, masks):
        images = images.to(self.device)
        masks = masks.to(self.device)
        output = self.model(images * masks.float(), masks.float())
        output = self.model(images * masks.float(), masks.float())
        loss = self.loss_function(output, images, masks)
        intersection, union = dice_coef(output, images, masks)
        return loss, intersection, union
    
    def save_weights(self):
        """Save the model weights to a file."""
        th.save(self.model.state_dict(), self.weights_path)
        
    def save_results(self, params_string: str = None, elapsed_time: float = None):
        """Save the training results to a file.

        Args:
            elapsed_time (float): elapsed time of the training
        """
        
        # Save the train losses to a txt file
        with open(self.results_path, 'w') as f:
            if elapsed_time is not None:
                f.write("Elapsed time [s]:\n")
                f.write(f"{elapsed_time}\n\n")
            f.write("Train losses\n")
            for loss in self.train_losses:
                f.write(f"{loss}\t")
            f.write("\n\n")
            f.write("Test losses\n")
            for loss in self.test_losses:
                f.write(f"{loss}\t")
            f.write("\n\n")
            f.write("Train dice coefficient\n")
            for dc in self.train_dice_coef:
                f.write(f"{dc}\t")
            f.write("\n\n")
            f.write("Test dice coefficient\n")
            for dc in self.test_dice_coef:
                f.write(f"{dc}\t")
            f.write("\n\n")
            f.write("Learning rate\n")
            for lr in self.training_lr:
                f.write(f"{lr}\t")
            f.write("\n\n")
            if params_string is not None:
                f.write("Parameters")
                f.write(params_string)
                f.write("\n\n")

            
            

if __name__ == "__main__":
    main()