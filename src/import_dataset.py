import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Subset
from torch.utils.data import DataLoader

def load_cifar10_data(normalize: bool =True, n_train: int = None, n_test: int = None, desired_classes: list=None) -> tuple:
    """Import the CIFAR-10 dataset and return the training and test datasets.

    Args:
        normalize (bool, optional): normalize, obtaining all values in the range [-1, 1]. Defaults to True.
        n_train (int, optional): number of training images, if None include them all. Defaults to None.
        n_test (int, optional): number of validation images, if None include them all. Defaults to None.
        desired_classes (list, optional): list of classes to include, if None include them all. Defaults to None.

    Returns:
        tuple: A tuple containing the training and test datasets.
    """
    # Define the transformation: convert images to tensors and normalize
    if normalize:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    else:
        # If no normalization is needed, just convert to tensor
        transform = transforms.ToTensor()

    # Load the training dataset
    trainset = torchvision.datasets.CIFAR10(root='../data', train=True, download=True, transform=transform)

    # Load the test dataset
    testset = torchvision.datasets.CIFAR10(root='../data', train=False, download=True, transform=transform)
    
    if desired_classes is None and n_train is None and n_test is None:
        # If no specific classes or counts are desired, return the full datasets
        return trainset, testset
    else:
        # Create a subset of the first 10,000 images
        if desired_classes is not None:
            # Filter the indices to include only the desired classes
            subset_indices_train = [i for i, (_, label) in enumerate(trainset) if label in desired_classes]
            subset_indices_test = [i for i, (_, label) in enumerate(testset) if label in desired_classes]
        
        if n_train is not None:
            # If no specific classes are desired, just take the first n_train images
            subset_indices_train = list(range(n_train))
        
        if n_test is not None:
            subset_indices_test = list(range(n_test))
        
        trainset = Subset(trainset, subset_indices_train)
        testset = Subset(testset, subset_indices_test)

        return trainset, testset

def create_dataloaders(trainset, testset, batch_size):
    # Create a DataLoader for the training dataset
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)

    # Create a DataLoader for the test dataset
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)

    return trainloader, testloader
