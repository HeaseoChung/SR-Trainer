from data.dataset import *


def define_dataset(common, dataset):
    dataset_name = dataset.type
    print(f"Dataset: {dataset_name} is going to be used")
    if dataset_name == "ImageDegradationDataset":
        return ImageDegradationDataset(common, dataset)
    elif dataset_name == "ImagePairDataset":
        return ImagePairDataset(common, dataset)
    elif dataset_name == "ImageDataset":
        return ImageDataset(common, dataset)
