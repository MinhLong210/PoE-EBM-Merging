from .eurosat import EuroSATBase
from .cars import Cars
from .dtd import DTD
from .mnist import MNIST
from .gtsrb import GTSRB
from .svhn import SVHN
from .sun397 import SUN397
from .resisc45 import RESISC45
from.cifar10 import CIFAR10Dataset
from .cifar100 import CIFAR100Dataset
from.fashionmnist import FashionMNISTDataset
from .flowers import Flowers102Dataset
from .food import Food101Dataset
from.pets import PetsDataset
# from .qnli import QNLI
# from .rte import RTE
# from .scitail import SCITAIL




eurosat = {
    'wrapper': EuroSATBase,
    'batch_size': 128,
    'res': 224,
    'type': 'eurosat',
    'num_workers': 8,
    'shuffle_train': True,
    'shuffle_test': False,
    'dir': './data/eurosat'
}

stanford_cars = {
    'wrapper': Cars,
    'batch_size': 128,
    'res': 224,
    'type': 'stanford_cars',
    'num_workers': 8,
    'shuffle_train': True,
    'shuffle_test': False,
    'dir': './data/stanford_cars'
}

mnist = {
    'wrapper': MNIST,
    'batch_size': 128,
    'res': 224,
    'type': 'mnist',
    'num_workers': 8,
    'shuffle_train': True,
    'shuffle_test': False,
    'dir': './data/mnist'
}

fashionmnist = {
    'wrapper': FashionMNISTDataset,
    'batch_size': 128,
    'res': 224,
    'type': 'fashionmnist',
    'num_workers': 8,
    'shuffle_train': True,
    'shuffle_test': False,
    'dir': './data/fashionmnist'
}

cifar10 = {
    'wrapper': CIFAR10Dataset,
    'batch_size': 128,
    'res': 224,
    'type': 'cifar10',
    'num_workers': 8,
    'shuffle_train': True,
    'shuffle_test': False,
    'dir': './data/cifar10'
}

cifar100 = {
    'wrapper': CIFAR100Dataset,
    'batch_size': 128,
    'res': 224,
    'type': 'cifar100',
    'num_workers': 8,
    'shuffle_train': True,
    'shuffle_test': False,
    'dir': './data/cifar100'
}

pets = {
    'wrapper': PetsDataset,
    'batch_size': 128,
    'res': 224,
    'type': 'pets',
    'num_workers': 8,
    'shuffle_train': True,
    'shuffle_test': False,
    'dir': './data/pets'
}

flowers = {
    'wrapper': Flowers102Dataset,
    'batch_size': 128,
    'res': 224,
    'type': 'flowers',
    'num_workers': 8,
    'shuffle_train': True,
    'shuffle_test': False,
    'dir': './data/flowers'
}

food = {
    'wrapper': Food101Dataset,
    'batch_size': 128,
    'res': 224,
    'type': 'food',
    'num_workers': 8,
    'shuffle_train': True,
    'shuffle_test': False,
    'dir': './data/food'
}


svhn = {
    'wrapper': SVHN,
    'batch_size': 128,
    'res': 224,
    'type': 'svhn',
    'num_workers': 8,
    'shuffle_train': True,
    'shuffle_test': False,
    'dir': './data/svhn'
}

dtd = {
    'wrapper': DTD,
    'batch_size': 128,
    'res': 224,
    'type': 'dtd',
    'num_workers': 8,
    'shuffle_train': True,
    'shuffle_test': False,
    'dir': './data/dtd'
}

sun397 = {
    'wrapper': SUN397,
    'batch_size': 128,
    'res': 224,
    'type': 'sun397',
    'num_workers': 8,
    'shuffle_train': True,
    'shuffle_test': False,
    'dir': './data/SUN397'
}

gtsrb = {
    'wrapper': GTSRB,
    'batch_size': 128,
    'res': 224,
    'type': 'gtsrb',
    'num_workers': 8,
    'shuffle_train': True,
    'shuffle_test': False,
    'dir': './data/gtsrb'
}

resisc45 = {
    'wrapper': RESISC45,
    'batch_size': 128,
    'res': 224,
    'type': 'resisc45',
    'num_workers': 8,
    'shuffle_train': True,
    'shuffle_test': False,
    'dir': './data/resisc45'
}

snli = {
    'dir': './datasets/snli',
    'type' : 'snli',
    'batch_size': 2,
    'num_workers': 16,
    'model_name_or_path' : "meta-llama/Meta-Llama-3-8B",
    'mask_class' : None,
}

mnli = {
    'dir': './datasets/mnli',
    'type' : 'mnli',
    'batch_size': 2,
    'num_workers': 16,
    'val_fraction': 0.2,
    'model_name_or_path' : "meta-llama/Meta-Llama-3-8B",
    'mask_class' : None,
    
}

sick = {
    'dir': './datasets/sick',
    'type' : 'sick',
    'batch_size': 2,
    'num_workers': 16,
    'model_name_or_path' : "meta-llama/Meta-Llama-3-8B",
    'mask_class' : None,
}

qnli = {
    'dir': './datasets/qnli',
    'type' : 'qnli',
    'batch_size': 2,
    'num_workers': 16,
    'val_fraction': 0.2,
    'model_name_or_path' : "meta-llama/Meta-Llama-3-8B",
    'mask_class' : 1,
}

rte = {
    'dir': './datasets/rte',
    'type' : 'rte',
    'batch_size': 2,
    'num_workers': 16,
    'val_fraction': 0.5,
    'model_name_or_path' : "meta-llama/Meta-Llama-3-8B",
    'mask_class' : 1,
}

scitail = {
    'dir': './datasets/scitail',
    'type' : 'scitail',
    'batch_size': 2,
    'num_workers': 16,
    'model_name_or_path' : "meta-llama/Meta-Llama-3-8B",
    'mask_class' : 2,
}


