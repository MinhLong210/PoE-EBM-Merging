import os

VIT_ARCH = 'ViT-L-14-CLIP'  # Model Architecture
MODEL_DIR = '../checkpoints_model_merging/FFT_checkpoints'               # Model Directory
PRETRAINED_MODEL_DIR = 'model-openai-clip-oc-vit-l-14'
CACHE_DIR = ''              # Where to cache HF pretrained checkpoints
HEAD_DIR = '../checkpoints_model_merging/heads'               # CLIP Head Directory
INF = 100000

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
        # 'name': 'hf_clip',
        # 'base_type': "openai/clip-vit-large-patch14",
        'name': 'oc_vit',
        'oc_name': 'ViT-L/14',
        'pretrained_name': 'ViT-L-14',
        'type': 'fft',
        'cachedir': CACHE_DIR,
        'pretrained_model_dir': PRETRAINED_MODEL_DIR,
        'bases': [
            # Path to model checkpoints stored locally - rank-16 8 vision tasks
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
            'target_modules_fft': ["in_proj_weight", "out_proj.weight"],
            'lora_dropout': 0.1,
            'bias': "none",
        },
    },
    'task_merge_config': {
        'representation': 'cauchy_ebm',
        'type': 'fft',
        'merge_type': 'mode',
        'jitter': 1e-3,
        'tol': 1e-2,
        'steps': 50,
        'scaling_coeffs': 1.0, #[.6],
        'arch': VIT_ARCH,
        
    },
    'eval_type': 'clip'
}

