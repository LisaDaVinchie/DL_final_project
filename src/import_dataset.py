import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Subset
from torch.utils.data import DataLoader

def load_cifar10_data(normalize=True, n_train = None, n_test = None, desired_classes=None):
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
    
    # Create a subset of the first 10,000 images
    if n_train is not None:
        if desired_classes is not None:
            # Filter the indices to include only the desired classes
            subset_indices = [i for i, (_, label) in enumerate(trainset) if label in desired_classes]
        else:
            # If no specific classes are desired, just take the first n_train images
            subset_indices = list(range(n_train))
        trainset = Subset(trainset, subset_indices)
    
    if n_test is not None:
        if desired_classes is not None:
            # Filter the indices to include only the desired classes
            subset_indices = [i for i, (_, label) in enumerate(testset) if label in desired_classes]
        else:
            # If no specific classes are desired, just take the first n_train images
            subset_indices = list(range(n_test))
        testset = Subset(testset, subset_indices)

    return trainset, testset

def create_dataloaders(trainset, testset, batch_size):
    # Create a DataLoader for the training dataset
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)

    # Create a DataLoader for the test dataset
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)

    return trainloader, testloader
