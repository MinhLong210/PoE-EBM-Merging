import os
import torch
import torchvision.datasets as datasets
from torchvision.datasets import OxfordIIITPet

ROOT = "../data"
class PetsDataset:
    def __init__(self,
                 is_train,
                 preprocess,
                 location=ROOT,
                 batch_size=128,
                 num_workers=16):

        split = "trainval" if is_train else "test"

        dataset = OxfordIIITPet(
            root=location,
            split=split,
            download=True,
            transform=preprocess,
            target_types="category"
        )

        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=is_train,
            num_workers=num_workers
        )

        if is_train:
            self.train_dataset = dataset
            self.train_loader = loader
        else:
            self.test_dataset = dataset
            self.test_loader = loader

        self.classnames = dataset.classes




def prepare_train_loaders(config):
    dataset_class = PetsDataset(
        is_train=True,
        preprocess=config['train_preprocess'],
        location=ROOT,
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
    )
    return {'full': dataset_class.train_loader}

def prepare_test_loaders(config):
    dataset_class = PetsDataset(
        is_train=False,
        preprocess=config['eval_preprocess'],
        location=ROOT,
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
    )

    loaders = {'test': dataset_class.test_loader}

    if config.get('val_fraction', 0) > 0.:
        test_set = loaders['test'].dataset
        shuffled_idxs = torch.load(config['shuffled_idxs'], weights_only=False)
        num_valid = int(len(test_set) * config['val_fraction'])
        valid_idxs, test_idxs = shuffled_idxs[:num_valid], shuffled_idxs[num_valid:]

        loaders['val'] = torch.utils.data.DataLoader(
            torch.utils.data.Subset(test_set, valid_idxs),
            batch_size=config['batch_size'],
            shuffle=False,
            num_workers=config['num_workers']
        )
        loaders['test'] = torch.utils.data.DataLoader(
            torch.utils.data.Subset(test_set, test_idxs),
            batch_size=config['batch_size'],
            shuffle=False,
            num_workers=config['num_workers']
        )

    loaders['class_names'] = dataset_class.classnames
    return loaders

