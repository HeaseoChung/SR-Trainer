from data.dataset import ImagePairDegradationDataset


def define_dataset(cfg):
    dataset_name = cfg.train.dataset.train.type
    print(f"Dataset: {dataset_name} is going to be used")
    if dataset_name == "ImagePairDegradationDataset":
        return ImagePairDegradationDataset(cfg)
    elif dataset_name == "ImagePairDataset":
        # TODO Add ImagePairDataset()
        pass
