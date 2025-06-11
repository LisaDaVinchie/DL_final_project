################################################################
#
#  Parameter optimization script, specific for lambda scheduler with
#  CIFAR-10 dataset and varying step size.
#  Keep batch size and epochs fixed, vary step size and
#  initial learning rate.
#
##################################################################


import optuna
import torch as th
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend
from train import TrainModel
from model import select_model
from losses import select_loss_function
from pathlib import Path
import argparse
from time import time
import json
import signal
import os
import sys
from import_dataset import create_dataloaders, load_cifar10_data

terminate_early = False
study = None
obj = None
terminate_early = False

def signal_handler(signum, frame):
    global terminate_early, study, obj
    print("Signal received, stopping optimization...", flush=True)
    terminate_early = True
    if study is not None:
        try:
            obj.save_optim_specs(study.best_trial, study)
            print("Best hyperparameters saved before exit.", flush=True)
        except Exception as e:
            print(f"Failed to save best trial: {e}", flush=True)
    sys.exit(0)


def main():
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
    
    
    results_path = Path(paths["next_result_path"])
    masks_path = Path(paths["current_mask_path"])
    masks_dir = masks_path.parent
        
    # Ensure the results file exists and is a txt file
    results_path = results_path.with_suffix('.txt')
    results_path.touch(exist_ok=True)
    
    training_params = params["training"]
    model_name = str(training_params["model_name"])
    loss_kind = str(training_params["loss_kind"])
    epochs = int(training_params["epochs"])
    batch_size = int(training_params["batch_size"])
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
    
    masks_path = masks_dir / f"mask_{mask_idx}.pt"
    
    if not masks_path.exists():
        raise FileNotFoundError(f"Masks file {masks_path} does not exist")
    
    global study, obj
    
    model = select_model(model_name)
    loss_function = select_loss_function(loss_kind)
    trainset, testset = load_cifar10_data(desired_classes=classes, n_train=n_train, n_test=n_test)
    masks = th.load(masks_path)
        
    obj = Objective(model=model,
                    loss_function=loss_function,
                    trainset = trainset,
                    testset = testset,
                    masks = masks,
                    epochs = epochs,
                    batch_size = batch_size)
    # Import parameters and paths
    obj.import_params(params)
    obj.import_and_check_paths(paths)
    
    storage = obj.create_storage()
    
    # Create a study to minimize the objective function
    study = optuna.create_study(direction="minimize",
                                storage=storage,
                                study_name=Path(obj.storage_path).stem,
                                load_if_exists=False)    

    
    # Register the signal handler
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler) # For SLURM jobs
    
    try:
        # Optimize in small chunks so we can handle early termination
        completed_trials = 0
        while not terminate_early and completed_trials < obj.n_trials:
            study.optimize(obj.objective, n_trials=1, gc_after_trial=True, catch=(Exception,))
            completed_trials += 1

            # Checkpoint: Save intermediate best trial
            print("Checkpoint: Saving intermediate best trial...", flush=True)
            obj.save_optim_specs(study.best_trial, study)

    except Exception as e:
        print(f"An error occurred during optimization: {e}", flush=True)
    finally:
        if study.best_trial is not None:
            obj.save_optim_specs(study.best_trial, study)
        print(f"\nElapsed time: {time() - start_time:.2f} seconds\n", flush=True)

class Objective():
    def __init__(self, model, loss_function, trainset, testset, masks, epochs, batch_size):
        # Load default config
        
        self.model = model
        self.loss_function = loss_function
        self.trainset = trainset
        self.testset = testset
        self.masks = masks
        self.epochs = epochs
        self.batch_size = batch_size
        
        self.dataloader_cache = {}
        
        self.optim_next_path = None

    def import_params(self, params: dict):
        self.params = params
        optim_params = params["optimization"]
        self.n_trials = optim_params["n_trials"]
        self.step_size_range = optim_params["step_size_range"]  # Default step size range
        self.learning_rate_range = optim_params["learning_rate_range"]  # Default learning rate range
        
    def import_and_check_paths(self, paths: Path):
        self.optim_next_path = Path(paths["next_optim_path"])
        self.storage_path = Path(paths["next_study_path"])
        self.dataset = None
        
        if not self.optim_next_path.parent.exists():
            raise FileNotFoundError(f"Optimization results dir {self.optim_next_path.parent} does not exist.")
        if not self.storage_path.parent.exists():
            raise FileNotFoundError(f"Storage path {self.storage_path} does not exist.")
    
    def create_storage(self, storage_path: Path = None):
        """Create a storage for Optuna."""
        if storage_path is None:
            storage_path = self.storage_path
        if not storage_path.parent.exists():
            raise FileNotFoundError(f"Storage path dir {storage_path.parent} does not exist.")
        return JournalStorage(JournalFileBackend(str(storage_path)))
        
    def objective(self, trial: optuna.Trial):
        # Suggest hyperparameters
        step_size = trial.suggest_int("step_size", self.step_size_range[0], self.step_size_range[1])
        learning_rate = trial.suggest_float("learning_rate", self.learning_rate_range[0], self.learning_rate_range[1])
        
        optimizer = th.optim.Adam(self.model.parameters(), lr=learning_rate)
        
        lr_lambda = lambda step: 2 ** -(step // step_size)
        scheduler = th.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
        
        train = TrainModel(model=self.model, 
                          optimizer=optimizer, 
                          loss_function=self.loss_function,
                          results_path = "",
                          weights_path="",
                          save_every=10000,
                          lr_scheduler = scheduler)
        
        train_loader, test_loader = create_dataloaders(self.trainset, self.testset, self.batch_size)
        
        train.train(train_loader, test_loader, self.masks, self.epochs)
        
        trial.set_user_attr("train_losses", train.train_losses)
        trial.set_user_attr("test_losses", train.test_losses)

        return train.test_losses[-1]  # Optuna minimizes this
    
    def save_optim_specs(self, trial, study: optuna.Study, optim_path: Path = None):
        # Save the best hyperparameters
        
        if optim_path is None:
            optim_path = self.optim_next_path
        
        if optim_path is None or not optim_path.parent.exists():
            raise FileNotFoundError(f"Optimization results path {optim_path} is not available.")
        
        json_str = json.dumps(self.params, indent=4)[1: -1]
        
        all_trials = []
        for t in study.trials:
            trial_info = {
                "number": t.number,
                "value": t.value,
                "params": t.params,
                "state": str(t.state),
                "start_time": str(t.datetime_start) if t.datetime_start else None,
                "end_time": str(t.datetime_complete) if t.datetime_complete else None,
                "user_attrs": t.user_attrs
            }
            all_trials.append(trial_info)
        
        with open(self.optim_next_path, "w") as f:
            f.write("Best trial parameters:\n")
            f.write(f"{json.dumps(trial.params, indent=4)}\n")
            f.write("\n")
            f.write("Best trial value:\n")
            f.write(f"{trial.value}\n")
            f.write("\n")
            f.write("Best trial train losses:\n")
            for i, loss in enumerate(trial.user_attrs["train_losses"]):
                f.write(f"{loss}\t")
            f.write("\n\n")
            f.write("Best trial test losses:\n")
            for i, loss in enumerate(trial.user_attrs["test_losses"]):
                f.write(f"{loss}\t")
            f.write("\n\n")
            f.write("All trials: \n")
            f.write(json.dumps(all_trials, indent=4))
            f.write("\n")
            f.write("Training parameters:\n")
            f.write(f"{json_str}\n")
                
            # Flush and sync to disk
            f.flush()
            os.fsync(f.fileno())
        
        print(f"Best hyperparameters saved to {optim_path}", flush = True)

if __name__ == "__main__":
    main()