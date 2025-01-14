class DatasetLoader:
    def __init__(self, dataset_name):
        self.dataset = load_dataset(dataset_name)

    def get_split(self, split="train"):
        return self.dataset[split]

# Example Usage
dataset_loader = DatasetLoader("truthful_qa")
train_dataset = dataset_loader.get_split("train")