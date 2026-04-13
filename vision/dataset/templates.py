####################### Vision Benchmark #######################

cars_template = [
    lambda c: f'a photo of a {c}.',
    lambda c: f'a photo of the {c}.',
    lambda c: f'a photo of my {c}.',
    lambda c: f'i love my {c}!',
    lambda c: f'a photo of my dirty {c}.',
    lambda c: f'a photo of my clean {c}.',
    lambda c: f'a photo of my new {c}.',
    lambda c: f'a photo of my old {c}.',
]

dtd_template = [
    lambda c: f'a photo of a {c} texture.',
    lambda c: f'a photo of a {c} pattern.',
    lambda c: f'a photo of a {c} thing.',
    lambda c: f'a photo of a {c} object.',
    lambda c: f'a photo of the {c} texture.',
    lambda c: f'a photo of the {c} pattern.',
    lambda c: f'a photo of the {c} thing.',
    lambda c: f'a photo of the {c} object.',
]

gtsrb_template = [
    lambda c: f'a zoomed in photo of a "{c}" traffic sign.',
    lambda c: f'a centered photo of a "{c}" traffic sign.',
    lambda c: f'a close up photo of a "{c}" traffic sign.',
]

mnist_template = [
    lambda c: f'a photo of the number: "{c}".',
]

eurosat_template = [
    lambda c: f'a centered satellite photo of {c}.',
    lambda c: f'a centered satellite photo of a {c}.',
    lambda c: f'a centered satellite photo of the {c}.',
]
resisc45_template = [
    lambda c: f'satellite imagery of {c}.',
    lambda c: f'aerial imagery of {c}.',
    lambda c: f'satellite photo of {c}.',
    lambda c: f'aerial photo of {c}.',
    lambda c: f'satellite view of {c}.',
    lambda c: f'aerial view of {c}.',
    lambda c: f'satellite imagery of a {c}.',
    lambda c: f'aerial imagery of a {c}.',
    lambda c: f'satellite photo of a {c}.',
    lambda c: f'aerial photo of a {c}.',
    lambda c: f'satellite view of a {c}.',
    lambda c: f'aerial view of a {c}.',
    lambda c: f'satellite imagery of the {c}.',
    lambda c: f'aerial imagery of the {c}.',
    lambda c: f'satellite photo of the {c}.',
    lambda c: f'aerial photo of the {c}.',
    lambda c: f'satellite view of the {c}.',
    lambda c: f'aerial view of the {c}.',
]

sun397_template = [
    lambda c: f'a photo of a {c}.',
    lambda c: f'a photo of the {c}.',
]

svhn_template = [
    lambda c: f'a photo of the number: "{c}".',
]

fashionmnist_template = [
    lambda c: f'a photo of a {c}.',
    lambda c: f'a photo of the {c}.',
    lambda c: f'a photo of a piece of {c}.',
    lambda c: f'a photo of my {c}.',
    lambda c: f'a close-up photo of a {c}.',
]

cifar_template = [
    lambda c: f'a photo of a {c}.',
    lambda c: f'a photo of the {c}.',
    lambda c: f'a blurry photo of a {c}.',
    lambda c: f'a close-up photo of a {c}.',
    lambda c: f'a photo of a small {c}.',
]

pets_template = [
    lambda c: f'a photo of a {c}.',
    lambda c: f'a photo of the {c}.',
    lambda c: f'a photo of a pet {c}.',
    lambda c: f'a close-up photo of a {c}.',
    lambda c: f'a photo of a cute {c}.',
]

flowers_template = [
    lambda c: f'a photo of a {c} flower.',
    lambda c: f'a photo of the {c} flower.',
    lambda c: f'a close-up photo of a {c} flower.',
    lambda c: f'a macro photo of a {c} flower.',
    lambda c: f'a photo of a blooming {c} flower.',
]

food_template = [
    lambda c: f'a photo of {c}.',
    lambda c: f'a photo of a plate of {c}.',
    lambda c: f'a close-up photo of {c}.',
    lambda c: f'a photo of freshly made {c}.',
    lambda c: f'a delicious photo of {c}.',
]


dataset_to_template = {
    'stanford_cars': cars_template,
    'dtd': dtd_template,
    'eurosat': eurosat_template,
    'gtsrb': gtsrb_template,
    'mnist': mnist_template,
    'resisc45': resisc45_template,
    'sun397': sun397_template,
    'svhn': svhn_template,
    'fashionmnist': fashionmnist_template,
    'cifar10': cifar_template,
    'cifar100': cifar_template,
    'pets': pets_template,
    'flowers': flowers_template,
    'food': food_template,
}


def get_templates(dataset_name):
    if dataset_name.endswith('Val'):
        return get_templates(dataset_name.replace('Val', ''))
    assert dataset_name in dataset_to_template, f'Unsupported dataset: {dataset_name}'
    return dataset_to_template[dataset_name]