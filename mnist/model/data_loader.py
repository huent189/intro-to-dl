import torchvision.datasets as datasets
from torchvision import transforms
import torch
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5), (0.5))])

def fetch_dataloader(types, params):
    """
    Fetches the DataLoader object for each type in types from data_dir.
    Args:
        types: (list) has one or more of 'train', 'val', 'test' depending on which data is required
        data_dir: (string) directory containing the dataset
        params: (Params) hyperparameters
    Returns:
        data: (dict) contains the DataLoader object for each type in types
    """
    dataloaders = {}

    if 'train' in types:
        train_raw = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        if 'val' in types:
            train_len = int(params.train_ratito * len(train_raw))
            train_raw, val_raw = torch.utils.data.random_split(train_raw, [train_len, len(train_raw)- train_len])
            dataloaders['val'] = torch.utils.data.DataLoader(val_raw, shuffle=True, batch_size=params.batch_size, num_workers=params.num_workers, pin_memory=params.cuda)
        dataloaders['train'] = torch.utils.data.DataLoader(train_raw, shuffle=True, batch_size=params.batch_size, num_workers=params.num_workers, pin_memory=params.cuda)
    if 'test' in types:
        test_raw = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
        dataloaders['test'] = torch.utils.data.DataLoader(test_raw, shuffle=True, batch_size=params.batch_size, num_workers=params.num_workers, pin_memory=params.cuda)
    return dataloaders