"""Functions to create dataloaders"""

from torch.utils.data import DataLoader


def create_dataloader(
    dataset,
    batch_size: int,
    num_workers: int,
    sampler=None,
    shuffle: bool = True,
    pin_memory: bool = True,
    drop_last: bool = False,
) -> DataLoader:
    """
    Construct Single or Multi modal dataloader, dependent on the dataset that is inputted

    Parameters
    ----------
    dataset:
        dataset for which a dataloader will be constructed
    batch_size: int
        number of samples used in each iteration
    num_workers: int
        number of processes used for multiprocessing
    sampler: Use a custom sampler, mutually exclusive with shuffle.
    shuffle: bool
        shuffle data points
    pin_memory: bool
        indicates whether the dataloader should enable memory pinning
    drop_last: bool
        indicates whether to drop the last batch, which may have unequal size to the rest

    Returns
    -------
    num_samples: int
        number of samples in dataloader
    dataloader: DataLoader
        dataloader associated to dataset
    """

    data_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        shuffle=shuffle,
        pin_memory=pin_memory,
        prefetch_factor=2,
        persistent_workers=True,
        drop_last=drop_last,
    )
    return data_loader
