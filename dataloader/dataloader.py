import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms
from load_data import load_labelled_and_unlabelled
from sklearn.model_selection import train_test_split
import os, sys
import numpy as np
import random


class Load_Dataset(Dataset):
    def __init__(self, dataset, dataset_configs):
        super().__init__()
        self.num_channels = dataset_configs.input_channels
        self.return_index = False

        # 1. Handle input format: Tuple (from prepare_dataset) or Dict (custom)
        if isinstance(dataset, dict):
            x_data = dataset["samples"]
            y_data = dataset.get("labels")
        elif isinstance(dataset, (tuple, list)):
            x_data = dataset[0]
            y_data = dataset[1] if len(dataset) > 1 else None
        else:
            raise TypeError(f"Dataset must be dict or tuple/list, got {type(dataset)}")

        # 2. Convert to Tensors
        if isinstance(x_data, np.ndarray):
            x_data = torch.from_numpy(x_data)
        
        if y_data is not None and isinstance(y_data, np.ndarray):
            y_data = torch.from_numpy(y_data)
        
        # 3. Fix Data Dimensions (N, C, L)
        if len(x_data.shape) == 2:
            x_data = x_data.unsqueeze(1)
        elif len(x_data.shape) == 3 and x_data.shape[1] != self.num_channels:
            x_data = x_data.permute(0, 2, 1)

        self.transform = None
        self.x_data = x_data.float()
        
        # 4. Efficiently Handle One-Hot Labels
        # Since data_pre_processing.py uses np.eye(), we know it's one-hot.
        if y_data is not None:
            # If (N, C) where C > 1 -> It's one-hot, use argmax
            if y_data.dim() == 2 and y_data.shape[1] > 1:
                y_data = torch.argmax(y_data, dim=1)
            # If (N, 1) -> Just squeeze
            elif y_data.dim() == 2 and y_data.shape[1] == 1:
                y_data = y_data.squeeze(1)
                
            self.y_data = y_data.long()
        else:
            self.y_data = None
            
        self.len = x_data.shape[0]         

    def __getitem__(self, index):
        x = self.x_data[index]
        if self.transform:
            x = self.transform(self.x_data[index].reshape(self.num_channels, -1, 1)).reshape(self.x_data[index].shape)
        y = self.y_data[index] if self.y_data is not None else None
        if self.return_index:
            return x, y, index
        else:
            return x, y


    def __len__(self):
        return self.len

def data_generator(data_path, dataset_configs, hparams, flag):
    """
    Load data from .pkl files using load_labelled_and_unlabelled function
    
    Args:
        data_path: Path to the dataset directory
        dataset_configs: Dataset configuration object
        hparams: Hyperparameters dictionary
        flag: 'source' or 'target'
        dtype: 'train', 'val', or 'test'
    """
    # Define activity mapping for harmonization
    activity_mapping = {
        'sitting': 'sitting',
        'Sitting and relaxing': 'sitting',
        'standing': 'standing',
        'Standing still': 'standing',
        'lying': 'lying',
        'Lying down': 'lying',
        'running': 'running',
        'Running': 'running',
        'walking': 'walking',
        'Walking': 'walking',
    }
    
    # Extract dataset name from path
    # dataset_name = os.path.basename(data_path)
    dataset_name = os.path.basename(data_path)
    base_dir = os.path.dirname(data_path)
    pkl_path = os.path.join(base_dir, f"{dataset_name}_processed.pkl")    
    # Load data using load_labelled_and_unlabelled
    prepared_datasets, _, _ = load_labelled_and_unlabelled(
        labelled_dataset_path=pkl_path,
        unlabelled_dataset_path=pkl_path if flag == 'target' else None,  # Use unlabeled for target
        activity_mapping=activity_mapping,
        verbose=0,
    )
    dataset_files=[]
    # Get the appropriate split based on flag
    if flag == 'source':
            # Source: Train, Test, Validation
        dataset_files.append(prepared_datasets['labelled']['train'])      # Index 0
        dataset_files.append(prepared_datasets['labelled']['test'])       # Index 1
        dataset_files.append(prepared_datasets['labelled']['val'])        # Index 2
    else:
        # Target: Unlabelled Train, Unlabelled Test (for adaptation)
        dataset_files.append(prepared_datasets['unlabelled']['train'])
        dataset_files.append(prepared_datasets['unlabelled']['train'])

    datasets=[Load_Dataset(dataset_files[x], dataset_configs) for x in range(len(dataset_files))]
    # Loading datasets
 
    # if dtype == "test":  # you don't need to shuffle or drop last batch while testing
    #     shuffle = False
    #     drop_last = False
    # else:
    shuffle = dataset_configs.shuffle
    drop_last = dataset_configs.drop_last

    # Dataloaders
    data_loaders=[]
    for idx, dataset in enumerate(datasets):
        
        data_loaders.append( torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=hparams["batch_size"],
            shuffle=False if idx==1 else True, #index of 1 is always the test set
            drop_last=False if idx==1 else True,
            num_workers=0
        ))

    return data_loaders

def data_generator_old(data_path, domain_id, dataset_configs, hparams):
    # loading path
    train_dataset = torch.load(os.path.join(data_path, "train_" + domain_id + ".pt"))
    test_dataset = torch.load(os.path.join(data_path, "test_" + domain_id + ".pt"))

    # Loading datasets
    train_dataset = Load_Dataset(train_dataset, dataset_configs)
    test_dataset = Load_Dataset(test_dataset, dataset_configs)

    # Dataloaders
    batch_size = hparams["batch_size"]
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size,
                                               shuffle=True, drop_last=True, num_workers=0)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size,
                                              shuffle=False, drop_last=dataset_configs.drop_last, num_workers=0)
    return train_loader, test_loader



def few_shot_data_generator(data_loader, dataset_configs, num_samples=5):
    x_data = data_loader.dataset.x_data
    y_data = data_loader.dataset.y_data

    NUM_SAMPLES_PER_CLASS = num_samples
    NUM_CLASSES = len(torch.unique(y_data))

    counts = [y_data.eq(i).sum().item() for i in range(NUM_CLASSES)]
    samples_count_dict = {i: min(counts[i], NUM_SAMPLES_PER_CLASS) for i in range(NUM_CLASSES)}

    samples_ids = {i: torch.where(y_data == i)[0] for i in range(NUM_CLASSES)}
    selected_ids = {i: torch.randperm(samples_ids[i].size(0))[:samples_count_dict[i]] for i in range(NUM_CLASSES)}

    selected_x = torch.cat([x_data[samples_ids[i][selected_ids[i]]] for i in range(NUM_CLASSES)], dim=0)
    selected_y = torch.cat([y_data[samples_ids[i][selected_ids[i]]] for i in range(NUM_CLASSES)], dim=0)

    few_shot_dataset = {"samples": selected_x, "labels": selected_y}
    few_shot_dataset = Load_Dataset(few_shot_dataset, dataset_configs)

    few_shot_loader = torch.utils.data.DataLoader(dataset=few_shot_dataset, batch_size=len(few_shot_dataset),
                                                  shuffle=False, drop_last=False, num_workers=0)

    return few_shot_loader

