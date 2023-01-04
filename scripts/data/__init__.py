from data.dataset import ImagePairDegradationDataset, ImagePairDataset

def define_dataset(common, dataset):
    dataset_name = dataset.type
    print(f"Dataset: {dataset_name} is going to be used")
    if dataset_name == "ImagePairDegradationDataset":
        return ImagePairDegradationDataset(common, dataset)
    elif dataset_name == "ImagePairDataset":
        return ImagePairDataset(common, dataset)
