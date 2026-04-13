import os

VIT_ARCH = 'ViT-L-14-CLIP'  # Model Architecture
CACHE_DIR = ''              # Where to cache HF pretrained checkpoints
MODEL_DIR = 'FFT_checkpoints'              # Model Directory
PRETRAINED_MODEL_DIR = 'model-openai-clip-oc-vit-l-14'
HEAD_DIR = 'heads'               # CLIP Head Directory
MODEL_DIR = '../checkpoints_model_merging/FFT_checkpoints'             # Model Directory
HEAD_DIR = '../checkpoints_model_merging/heads'

config = {
    'dataset': [
        {
            'name': 'svhn',
            'shuffle_train': True,
            'crop_ratio': 1.0,
            'clip_encodings': os.path.join(HEAD_DIR, VIT_ARCH, 'svhn_head.pt'),
            'val_fraction': 0.4,
            'batch_size': 32,
            'num_workers': 8,
            'shuffled_idxs': os.path.join(os.getcwd(), 'dataset/shuffled_idxs/svhn_shuffled_idxs.pt')
        },
        {
            'name': 'mnist',
            'shuffle_train': True,
            'crop_ratio': 1.0,
            'clip_encodings': os.path.join(HEAD_DIR, VIT_ARCH, 'mnist_head.pt'),
            'val_fraction': 0.2,
            'batch_size': 32,
            'num_workers': 8,  
            'shuffled_idxs': os.path.join(os.getcwd(), 'dataset/shuffled_idxs/mnist_shuffled_idxs.pt')
        },

        
        
        {
            'name': 'stanford_cars',
            'shuffle_train': True,
            'crop_ratio': 1.0,
            'clip_encodings': os.path.join(HEAD_DIR, VIT_ARCH, 'stanford_cars_head.pt'),
            'val_fraction': 0.2,
            'batch_size': 32,
            'num_workers': 16,
            'shuffled_idxs': os.path.join(os.getcwd(), 'dataset/shuffled_idxs/cars_shuffled_idxs.pt')
        },
        {
            'name': 'dtd',
            'shuffle_train': True,
            'crop_ratio': 1.0,
            'clip_encodings': os.path.join(HEAD_DIR, VIT_ARCH, 'dtd_head.pt'),
            'batch_size': 32,
            'num_workers': 16,
        },
        {
            'name': 'eurosat',
            'shuffle_train': True,
            'crop_ratio': 1.0,
            'clip_encodings': os.path.join(HEAD_DIR, VIT_ARCH, 'eurosat_head.pt'),
            'batch_size': 32,
            'num_workers': 16,
        },
        {
            'name': 'gtsrb',
            'shuffle_train': True,
            'crop_ratio': 1.0,
            'clip_encodings': os.path.join(HEAD_DIR, VIT_ARCH, 'gtsrb_head.pt'),
            'val_fraction': 0.2,
            'batch_size': 32,
            'num_workers': 16,
            'shuffled_idxs': os.path.join(os.getcwd(), 'dataset/shuffled_idxs/gtsrb_shuffled_idxs.pt')
        },
        
        {
            'name': 'resisc45',
            'shuffle_train': True,
            'crop_ratio': 1.0,
            'clip_encodings': os.path.join(HEAD_DIR, VIT_ARCH, 'resisc45_head.pt'),
            'batch_size': 32,
            'num_workers': 16,
        },

        # {
        #     'name': 'cifar10',
        #     'shuffle_train': True,
        #     'crop_ratio': 1.0,
        #     'clip_encodings': os.path.join(HEAD_DIR, VIT_ARCH, 'cifar10_head.pt'),
        #     'batch_size': 32,
        #     'num_workers': 16,
        # },

        # {
        #     'name': 'cifar100',
        #     'shuffle_train': True,
        #     'crop_ratio': 1.0,
        #     'clip_encodings': os.path.join(HEAD_DIR, VIT_ARCH, 'cifar100_head.pt'),
        #     'batch_size': 32,
        #     'num_workers': 16,
        # },

        # {
        #     'name': 'fashionmnist',
        #     'shuffle_train': True,
        #     'crop_ratio': 1.0,
        #     'clip_encodings': os.path.join(HEAD_DIR, VIT_ARCH, 'fashionmnist_head.pt'),
        #     'batch_size': 32,
        #     'num_workers': 16,
        # },

        # {
        #     'name': 'flowers',
        #     'shuffle_train': True,
        #     'crop_ratio': 1.0,
        #     'clip_encodings': os.path.join(HEAD_DIR, VIT_ARCH, 'flowers_head.pt'),
        #     'batch_size': 32,
        #     'num_workers': 16,
        # },

        # {
        #     'name': 'food',
        #     'shuffle_train': True,
        #     'crop_ratio': 1.0,
        #     'clip_encodings': os.path.join(HEAD_DIR, VIT_ARCH, 'food_head.pt'),
        #     'batch_size': 32,
        #     'num_workers': 16,
        # },

        # {
        #     'name': 'pets',
        #     'shuffle_train': True,
        #     'crop_ratio': 1.0,
        #     'clip_encodings': os.path.join(HEAD_DIR, VIT_ARCH, 'pets_head.pt'),
        #     'batch_size': 32,
        #     'num_workers': 16,
        # }
    ],
    'model': {
        'name': 'oc_vit',
        'oc_name': 'ViT-L/14',
        'pretrained_name': 'ViT-L-14',
        # 'base_type': "openai/clip-vit-large-patch14",
        'cachedir': CACHE_DIR,
        'pretrained_model_dir': PRETRAINED_MODEL_DIR,
        'bases': [
            f'{MODEL_DIR}/{VIT_ARCH}/CarsVal/nonlinear_finetuned.pt',
            f'{MODEL_DIR}/{VIT_ARCH}/DTDVal/nonlinear_finetuned.pt',
            f'{MODEL_DIR}/{VIT_ARCH}/EuroSATVal/nonlinear_finetuned.pt',
            f'{MODEL_DIR}/{VIT_ARCH}/GTSRBVal/nonlinear_finetuned.pt',
            f'{MODEL_DIR}/{VIT_ARCH}/MNISTVal/nonlinear_finetuned.pt',
            f'{MODEL_DIR}/{VIT_ARCH}/RESISC45Val/nonlinear_finetuned.pt',
            f'{MODEL_DIR}/{VIT_ARCH}/SVHNVal/nonlinear_finetuned.pt',

            # f'{MODEL_DIR}/{VIT_ARCH}/CIFAR10Val/nonlinear_finetuned.pt',
            # f'{MODEL_DIR}/{VIT_ARCH}/CIFAR100Val/nonlinear_finetuned.pt',
            # f'{MODEL_DIR}/{VIT_ARCH}/FashionMNISTVal/nonlinear_finetuned.pt',
            # f'{MODEL_DIR}/{VIT_ARCH}/Flowers102Val/nonlinear_finetuned.pt',
            # f'{MODEL_DIR}/{VIT_ARCH}/Food101Val/nonlinear_finetuned.pt',
            # f'{MODEL_DIR}/{VIT_ARCH}/OxfordIIITPetVal/nonlinear_finetuned.pt',
            
            
        ],
        'ft_config': {
            'type': 'fft',
            'r': 16,
            'lora_alpha': 16,
            'target_modules': ["q_proj", "k_proj", "v_proj", "out_proj"],
            'lora_dropout': 0.1,
            'bias': "none",
        },
    },
    'task_merge_config': {
        'representation': 'vector',
        'scaling_coeffs': 1.0,
        'corrupt': False,
        'merge_method': 'tv',
        'merging_type': 'mean',
        'dare' : False,
        'dare_pruning_coeffs' : 0.0,
    },
    'eval_type': 'clip',
}

