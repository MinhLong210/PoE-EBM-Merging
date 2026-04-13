from collections import defaultdict, OrderedDict
import torch.nn.functional as F
from tqdm.auto import tqdm
from copy import deepcopy
from time import time

import multiprocessing as mp
# mp.set_start_method('spawn', force=True)
import os

from torch import nn
import torch
import pdb
from joblib import Parallel, delayed
from utils import get_merging_fn, get_mask_fn, compute_norm, preprocess_ftms_to_base_sd
from masking_ops import masked_merge
import json

class VectorOps(nn.Module):
    def directions_to_reps(self, directions):
        if isinstance(directions, list):
            return [self.directions_to_reps(direction) for direction in directions]
        return torch.nn.utils.parameters_to_vector(
            [value.reshape(-1) for key, value in directions.items()]
        )

    def rep_to_state_dict(self, vector, state_dict, remove_keys=[]):
        if isinstance(vector, list) or len(vector.shape) == 2:
            return [self.rep_to_state_dict(v, state_dict, remove_keys) for v in vector]
        # create a reference dict to define the order of the vector
        reference_dict = deepcopy(state_dict)
        for key in remove_keys:
            if key in reference_dict:
                del reference_dict[key]
        sorted_reference_dict = OrderedDict(sorted(reference_dict.items()))

        # create a shared state dict using the refence dict
        torch.nn.utils.vector_to_parameters(vector, sorted_reference_dict.values())

        # add back the encoder and decoder embedding weights.
        if "transformer.shared.weight" in sorted_reference_dict:
            for key in remove_keys:
                sorted_reference_dict[key] = sorted_reference_dict[
                    "transformer.shared.weight"
                ]
        return sorted_reference_dict

    def mask_to_state_dict(self, mask, state_dict, remove_keys=[]):
        if isinstance(mask, list):
            return [self.mask_to_state_dict(m, state_dict, remove_keys) for m in mask]
        return self.rep_to_state_dict(mask, state_dict, remove_keys)

    def forward(self, directions, merging_fn, merge_config):
        vectors = self.directions_to_reps(directions)
        merged_vector,rows_to_keep, topk_mask = merging_fn(vectors)
        mask_sd = self.rep_to_state_dict(topk_mask, directions[0])

        ties_mask = [dict() for _ in range(len(rows_to_keep))]
        for idx in range(len(rows_to_keep)):
            ties_mask[idx] = self.rep_to_state_dict(rows_to_keep[idx], directions[0])
        sd = self.rep_to_state_dict(merged_vector, directions[0])

        return sd, ties_mask


class TaskMerger(nn.Module):
    def __init__(self, finetuned_models, pretrained_model, param_handler, device=0, merge_config=None):
        super().__init__()

        self.device = device
        self.scaling_coeffs = torch.tensor([1.] * len(finetuned_models))
        self.param_handler = param_handler
        self.finetuned_models = finetuned_models
        self.ftms_params = [param_handler(ft_model) for ft_model in finetuned_models]
        self.pretrained_model = pretrained_model.cpu()
        self.pt_params = self.pretrained_model.state_dict()
        self.merge_config = merge_config


    # def randbin(self, M, N, P):
    #     P = 1-P
    #     return torch.randint(2, size=(M, N), dtype=torch.float32).bernoulli(P)
    def randbin(self, shape, p, device=None, dtype=torch.float32):
        """
        Generate inverted Bernoulli mask with P(mask=1)=1-p
        """
        keep_prob = 1.0 - p
        return torch.bernoulli(
            torch.full(shape, keep_prob, device=device, dtype=dtype)
        )

    def apply_dare(self, ftms_params, p, dare_seed = 0):
        print("DARE seed: ", dare_seed)
        torch.manual_seed(dare_seed)
        finetuned_directions = []
        for ftm_params in ftms_params:
            direction_sd = {}
            for key, finetuned_val in ftm_params.items():
                try:
                    # direction_sd[key] = finetuned_val * self.randbin(finetuned_val.shape[0], finetuned_val.shape[1], p) * (1/(1-p))
                    mask = self.randbin(finetuned_val.shape, p, device=finetuned_val.device)
                    direction_sd[key] = finetuned_val * mask / (1.0 - p)
                except:
                    pdb.set_trace()
            finetuned_directions += [OrderedDict(sorted(direction_sd.items()))]
        return finetuned_directions

    def get_task_directions(self, ptm_params, ftms_params): # Task_vector = finetuned - pretrained
        finetuned_directions = []
        for ftm_params in ftms_params:
            direction_sd = {}

            for key, finetuned_val in ftm_params.items():
                if key not in ptm_params:
                    ptm_val = torch.zeros_like(finetuned_val)
                else:
                    ptm_val = ptm_params[key]

                direction_sd[key] = finetuned_val - ptm_val
            finetuned_directions += [OrderedDict(sorted(direction_sd.items()))]
        return finetuned_directions

    def set_scaling_coeffs(self, scaling_coeffs):
        if isinstance(scaling_coeffs, float) or len(scaling_coeffs) == 1:
            self.scaling_coeffs = torch.tensor([scaling_coeffs] * len(self.ftms_params))
        else:
            self.scaling_coeffs = torch.tensor(scaling_coeffs)

    # def get_layer_names(self, state_dict):
    #     layer_names = defaultdict(lambda: dict())
    #     for key in state_dict:
    #         if ('.weight' in key) or ('_weight' in key):
    #             strip_key = key.replace('.weight', '').replace('_weight', '')
    #             layer_names[strip_key]['weight'] = key
    #         elif ('.bias' in key) or ('_bias' in key):
    #             strip_key = key.replace('.bias', '').replace('_bias', '')
    #             layer_names[strip_key]['bias'] = key
    #         else:
    #             layer_names[key]['other'] = key + ':other'
    #     return layer_names

    def get_layer_names(self, state_dict):
        layer_names = defaultdict(dict)          # ← dict is a built-in callable → perfectly picklable
        # or equivalently:
        # layer_names = defaultdict(lambda: {})  # ← this also works reliably in most cases

        for key in state_dict:
            if '.weight' in key or '_weight' in key:
                strip_key = key.replace('.weight', '').replace('_weight', '')
                layer_names[strip_key]['weight'] = key
            elif '.bias' in key or '_bias' in key:
                strip_key = key.replace('.bias', '').replace('_bias', '')
                layer_names[strip_key]['bias'] = key
            else:
                layer_names[key]['other'] = key + ':other'

        return layer_names

    def add_task_parameters(self, base_model, parameters, concat_across_output = True, scaling_coeffs=1.):
        if isinstance(parameters, list):
            return [self.add_task_parameters(
                deepcopy(base_model),
                parameter,
                concat_across_output=concat_across_output,
                scaling_coeffs=scaling_coeffs
            ) for parameter in parameters]
        sd = base_model.state_dict()
        for key, val in parameters.items():
            try:
                if (concat_across_output):
                    sd[key].add_(val.cpu() * scaling_coeffs)
                else:
                    sd[key].add_(val.T.cpu() * scaling_coeffs)
            except:

                splitted_key_components = key.split(".")
                if splitted_key_components[-1] == "weight":
                    splitted_key_components.append("base_layer")
                else:
                    splitted_key_components.append("weight")
                    splitted_key_components.append("base_layer")

                splitted_key_components[-2], splitted_key_components[-1] = splitted_key_components[-1], splitted_key_components[-2]
                key = ".".join(splitted_key_components)
                if (concat_across_output):
                    sd[key].add_(val.cpu() * scaling_coeffs)
                else:
                    sd[key].add_(val.T.cpu() * scaling_coeffs)
        return base_model

    def add_task_parameters_fft(self, base_model, parameters, concat_across_output = True, scaling_coeffs=1.):
        if isinstance(parameters, list):
            return [self.add_task_parameters(
                deepcopy(base_model),
                parameter,
                concat_across_output=concat_across_output,
                scaling_coeffs=scaling_coeffs
            ) for parameter in parameters]
        sd = base_model.state_dict()
        for key, val in parameters.items():
            if (concat_across_output):
                try:
                    sd[key].add_(val.cpu() * scaling_coeffs)
                except:
                    import pdb; pdb.set_trace()
            else:
                sd[key].add_(val.T.cpu() * scaling_coeffs)

        return base_model

    def directions_to_matrices(self, directions, reference_layer_names=None):
        if isinstance(directions, list):
            return [self.directions_to_matrices(direction, reference_layer_names) for direction in directions]

        if reference_layer_names is None:
            layer_names = self.get_layer_names(directions)
        else:
            layer_names = reference_layer_names

        matrices = {}
        for layer_name, parameter_names in layer_names.items():
            if 'other' in parameter_names:
                other_parameter = directions[parameter_names['other'].replace(':other', '')].to(torch.float32)
                # Ensure parameters are always two dimensional
                if len(other_parameter.shape) == 1: # e.g., class token, positional embeddings
                    other_parameter = other_parameter[None, :]
                elif len(other_parameter.shape) > 2: # e.g., patch embeddings
                    other_parameter = other_parameter.flatten(1)
                matrices[layer_name + ':other'] = other_parameter
            elif 'weight' in parameter_names:
                weight_name = parameter_names['weight']
                weight = directions[weight_name]
                if 'norm' in layer_name or 'ln' in layer_name:
                    weight = torch.diag(weight)
                matrices[layer_name] = weight.flatten(1)
                if 'bias' in parameter_names:
                    bias = directions[parameter_names['bias']]
                    matrices[layer_name] = torch.concat((matrices[layer_name], bias.reshape(-1, 1)), dim=1)
        return matrices

    def matrix_to_state_dict(self, matrix, state_dict, remove_keys=[]):
        if isinstance(matrix, list):
            return [self.matrix_to_state_dict(m, state_dict) for m in matrix]

        reference_dict = deepcopy(state_dict)
        for key in remove_keys:
            if key in reference_dict:
                del reference_dict[key]

        layer_names = self.get_layer_names(reference_dict)
        merged_state_dict = {}
        for layer_name, value in matrix.items():
            try:
                parameter_types = layer_names[layer_name.replace(':other', '')]
                if 'other' in parameter_types:
                    name = parameter_types['other'].replace(':other', '')
                    merged_state_dict[name] = value.reshape(reference_dict[name].shape)
                else:
                    if 'bias' in parameter_types:
                        bias_index = value.shape[1] - 1
                        value, bias = value[:, :bias_index], value[:, -1].flatten()
                        merged_state_dict[parameter_types['bias']] = bias
                    if 'norm' in layer_name or 'ln' in layer_name:
                        value = torch.diagonal(value)
                    name = parameter_types['weight']
                    merged_state_dict[name] = value.reshape(*(reference_dict[name].shape))
            except:
                pdb.set_trace()

        # add back the encoder and decoder embedding weights.
        if "transformer.shared.weight" in merged_state_dict:
            for key in remove_keys:
                merged_state_dict[key] = merged_state_dict[
                    "transformer.shared.weight"
                ]
        return merged_state_dict

    def transform(self, *args, **kwargs):
        return

class VectorMerger(TaskMerger):
    def __init__(self, finetuned_models, pretrained_model, param_handler, device=0, merge_config=None):
        super().__init__(
            finetuned_models=finetuned_models,
            pretrained_model=pretrained_model,
            param_handler=param_handler,
            device=device,
            merge_config=merge_config
        )

        self.representation_helper = VectorOps()

    def merge(self, merge_config={'merge_method': 'tv'}):
        print(merge_config['merge_method'])
        merging_fn = lambda x: get_merging_fn(merge_config['merge_method'])(
            x, **merge_config, weights=self.scaling_coeffs
        )

        ptm_reference_params = self.param_handler(self.pretrained_model).get_ft_parameters()
        ftms_relevant_params = [ftm.get_ft_parameters() for ftm in self.ftms_params]
        ftms_task_dirs = self.get_task_directions(ptm_reference_params, ftms_relevant_params)

        if merge_config.get('dare', False):
            ftms_task_dirs = self.apply_dare(
                ftms_task_dirs, merge_config['dare_pruning_coeffs'], merge_config['dare_seed']
            )
        merged_sd = self.representation_helper(ftms_task_dirs, merging_fn, merge_config)
        merged_base = deepcopy(self.pretrained_model)
        if len(merged_sd) == 2:
            merged_sd, mask = merged_sd


        deviations = compute_deviations(ftms_task_dirs, merged_sd)
        # with open(f"logs/LoRA_ViTB32/deviations_DARE_TIES{self.merge_config['scaling_coeffs']}.json", "w") as f:
        #     json.dump(deviations, f)
        # import pdb; pdb.set_trace()


        merged_model = self.add_task_parameters(merged_base, merged_sd)
        import pdb; pdb.set_trace()

        # Save merged vector
        # name = "TV_LoRA"
        # torch.save(merged_sd, "logs/ViT-L-14" + name + ".pt")

        return merged_model


class CauchyEBMerger(TaskMerger):
    def __init__(self, finetuned_models, pretrained_model, param_handler, device=0, merge_config=None):
        super().__init__(
            finetuned_models=finetuned_models,
            pretrained_model=pretrained_model,
            param_handler=param_handler,
            device=device,
            merge_config=merge_config
        )

        self.layer_names = self.get_layer_names(self.ftms_params[0].get_ft_parameters())
        self.representation_helper = VectorOps()
        self.ingredients = None
        self.changes_dict = {}

    ############################### Helpers
    def variable_extend_dim(self, elements, op_dim):
        if isinstance(elements, list):
            return [self.variable_extend_dim(element, op_dim) for element in elements]
        while len(elements.shape) < (op_dim+1):
            elements = elements.unsqueeze(-1)
        return elements

    def dict_of_concat_matrices(self, list_of_dictmatrices, dim=0, concat_across_output = True):
        dict2matrix_stack = defaultdict(lambda: list())
        for dict2matrix in list_of_dictmatrices:
            for key, val in dict2matrix.items():
                if(concat_across_output == True):
                    dict2matrix_stack[key] += [val.to(self.device)]
                else:
                    dict2matrix_stack[key] += [val.T.to(self.device)]

        for key, list_of_vals in dict2matrix_stack.items():
            # Extend dim as necessary
            list_of_vals = self.variable_extend_dim(list_of_vals, op_dim=dim)
            dict2matrix_stack[key] = torch.concat(list_of_vals, dim=dim)
        return dict2matrix_stack

    def remove_others(self, ftms_mats):
        other_mats = [dict() for i in range(len(ftms_mats))]
        transform_mats = [dict() for i in range(len(ftms_mats))]

        for m_idx, ftm_mats in enumerate(ftms_mats):
            for key, val in ftm_mats.items():
                if ':other' in key:
                    other_mats[m_idx][key] = val.to(self.device)
                elif 'modules_to_save' in key:
                    other_mats[m_idx][key] = val.to(self.device)
                else:
                    transform_mats[m_idx][key] = val.to(self.device)
        print(f'Len other: {len(other_mats[0])}| len: transform: {len(transform_mats[0])}')
        return other_mats, transform_mats

    def add_others(self, ftms_mats, ftms_others):
        if isinstance(ftms_mats, list):
            return [self.add_others(ftms_mat, ftms_other) for ftms_mat, ftms_other in zip(ftms_mats, ftms_others)]

        for key, val in ftms_others.items():
            ftms_mats[key] = val
        return ftms_mats
    
    def get_merged_task_vectors_cauchy_merge_mode_LoRA(self, ftms_mats, max_iters=50, tol=1e-5, jitter=1e-3):
        layers = ftms_mats[0].keys()
        merged_weights = {}
        num_tasks = len(ftms_mats)

        for layer_key in tqdm(layers, desc="Iterative closed-form merging"):
            # If not matrix then just average
            if len(ftms_mats[0][layer_key].shape) != 2:
                merged_weights[layer_key] = torch.stack([ftms_mats[i][layer_key] for i in range(num_tasks)], dim=0).mean(dim=0)
                print(f"Averaging for layer {layer_key} with shape {ftms_mats[0][layer_key].shape}")
                continue

            # Stack all task matrices: (N, dim, dim)
            A_stack = torch.stack([ftms_mats[i][layer_key] for i in range(num_tasks)], dim=0)
            A_stack = A_stack.to(self.device)
            N, dim, _ = A_stack.shape


            # Initial guess: uniform merge using original method
            A_T = A_stack.transpose(1, 2)
            ATA_stack = torch.bmm(A_stack, A_T)                  # (N, d, d)
            H_init = ATA_stack.sum(dim=0) + jitter * torch.eye(dim).to(ATA_stack.device) # (d, d)
            ATATA_stack = torch.bmm(ATA_stack, A_stack)          # (N, d, d)
            # ||A_i||_F^2
            task_norm_sq = (A_stack ** 2).sum(dim=(1, 2))  # (N,)
            b_init = ATATA_stack.sum(dim=0)                     # (d, d)
            # merged = torch.linalg.pinv(H_init) @ b_init          # initial merged
            L_init = torch.linalg.cholesky(H_init)
            merged = torch.cholesky_solve(b_init, L_init)

            for iter_idx in range(max_iters):
                # Compute diff = merged - A_i 
                diff = merged - A_stack 

                # Compute A_i^T @ diff
                A_T_diff = torch.bmm(A_stack.transpose(1, 2), diff)

                # Denom: ||A_i^T @ diff||_F^2 + eps
                denom = (A_T_diff ** 2).sum(dim=(1, 2)) +  tol * task_norm_sq
                # denom = (A_T_diff ** 2).sum(dim=(1, 2)) +  tol

                alpha = 1.0 / denom  # (N,)

                # Compute weighted H = sum alpha_i * (A_i @ A_i^T) 
                weighted_ATA = ATA_stack * alpha.view(-1, 1, 1)      # (N, d, d) * (N,1,1)
                H = weighted_ATA.sum(dim=0) + jitter * torch.eye(dim).to(ATA_stack.device) # (d, d)

                # Compute weighted b = sum alpha_i * (A_i @ A_i^T @ A_i) 
                weighted_ATATA = ATATA_stack * alpha.view(-1, 1, 1)
                b = weighted_ATATA.sum(dim=0)                        # (d, d)

                L = torch.linalg.cholesky(H)
                new_merged = torch.cholesky_solve(b, L)

                change = torch.norm(new_merged - merged, p='fro').item()

                merged = new_merged

                if change < 1e-5:
                    print(f"Layer {layer_key}: converged at iter {iter_idx+1} (change={change:.6f})")
                    break
            else:
                print(f"Layer {layer_key}: max iters reached (final change={change:.6f})")


            merged_weights[layer_key] = merged

        return merged_weights

    def get_merged_task_vectors_cauchy_merge_FFT(self, ftms_mats, max_iters=50, tol=1e-5):
        new_ftms_mats = []
        if self.merge_config["type"] == "fft":
            # Handle ftms_mats keys
            for i in range(len(ftms_mats)):
                clean_sd = preprocess_ftms_to_base_sd(ftms_mats[i], arch=self.merge_config["arch"])
                new_ftms_mats.append(clean_sd)
            ftms_mats = new_ftms_mats

        layers = ftms_mats[0].keys()
        merged_weights = {}
        num_tasks = len(ftms_mats)

        for layer_key in tqdm(layers, desc="Iterative closed-form merging"):
            print("Layer:", layer_key)
            W0 = ftms_mats[0][layer_key]

            # If bias then just average
            if len(W0.shape) != 2:
                merged_weights[layer_key] = torch.stack([ftms_mats[i][layer_key] for i in range(num_tasks)], dim=0).mean(dim=0)
                continue

            # If weights
            if W0.ndim == 2: 
                if not any(s in layer_key for s in ("ln_post", "conv", "c_proj", "ln_pre")):
                    d = W0.shape[1]  # model dim (e.g. 768)

                    num_chunks = W0.shape[0] // d

                    merged_chunks = []

                    for chunk_idx in range(num_chunks):
                        row_slice = slice(chunk_idx * d, (chunk_idx + 1) * d)

                        chunk_mats_sq = [
                            ftms_mats[i][layer_key][row_slice, :]   # (d, d)
                            for i in range(num_tasks)
                        ]

                        merged_sq, changes_dict = self.cauchy_ebm_merge_mode(
                            chunk_mats_sq,
                            max_iters=self.merge_config['steps'],
                            jitter=self.merge_config['jitter'],
                            tol=self.merge_config['tol'],
                            device=self.device,
                            merge_config=self.merge_config
                        )  # (d, d)

                        merged_chunks.append(merged_sq)

                        self.changes_dict[f"{layer_key}"] = changes_dict
                    # Reassemble (n*d, d)
                    try:
                        merged_weights[layer_key] = torch.cat(merged_chunks, dim=0)

                    except:
                        import pdb; pdb.set_trace()
                    

                else: # average for conv and ln
                    avg_layers_list = [
                        ftms_mats[i][layer_key] for i in range(num_tasks)
                    ]
                    merged_weights[layer_key] = torch.stack(avg_layers_list, dim=0).mean(dim=0)

        return merged_weights

    def cauchy_ebm_merge_mode(self, A_list, max_iters, jitter, tol, device, merge_config):
        num_tasks = len(A_list)
        A_stack = torch.stack(A_list, dim=0)
        A_stack = A_stack.to(device)
        N, dim, _ = A_stack.shape

        A_stack = A_stack.to(device)
        N, dim, _ = A_stack.shape

        # Initialization
        A_T = A_stack.transpose(1, 2)
        ATA_stack = torch.bmm(A_stack, A_T)                  # (N, d, d)
        # ||A_i||_F^2
        task_norm_sq = (A_stack ** 2).sum(dim=(1, 2))  # (N,)
        # H_init = ATA_stack.sum(dim=0)
        H_init = ATA_stack.sum(dim=0) + jitter * torch.eye(dim).to(ATA_stack.device)                    # (d, d)
        ATATA_stack = torch.bmm(ATA_stack, A_stack)          # (N, d, d)
        b_init = ATATA_stack.sum(dim=0)                     # (d, d)
        L_init = torch.linalg.cholesky(H_init)
        merged = torch.cholesky_solve(b_init, L_init)

        changes_dict = []
        for iter_idx in range(max_iters):
            diff = merged - A_stack  # broadcasting
            A_T_diff = torch.bmm(A_stack.transpose(1, 2), diff)
            quad_term = (A_T_diff ** 2).sum(dim=(1, 2))  # (N,)


            # denom = ((θ_i − θ_m)^T θ_i)^2 + tol * ||θ_i||^2
            denom = quad_term + tol * task_norm_sq

            # alpha_i = 1 / denom
            alpha = 1.0 / denom  # (N,)

            # Compute weighted H = sum alpha_i * (A_i @ A_i^T)
            weighted_ATA = ATA_stack * alpha.view(-1, 1, 1)      # (N, d, d) * (N,1,1)
            H = weighted_ATA.sum(dim=0) + jitter * torch.eye(dim).to(weighted_ATA.device)   # (d, d)

            # Compute weighted b = sum alpha_i * (A_i @ A_i^T @ A_i)
            weighted_ATATA = ATATA_stack * alpha.view(-1, 1, 1)
            b = weighted_ATATA.sum(dim=0)                        # (d, d)

            L = torch.linalg.cholesky(H)
            new_merged = torch.cholesky_solve(b, L)

            change = torch.norm(new_merged - merged, p='fro').item()
            if change < 1e-5:
                print(f"converged at iter {iter_idx+1} (change={change:.6f})")
                break
            changes_dict.append(change)

            merged = new_merged


        return merged, changes_dict


    def transform(self, merge_config):
        # Setup parameters
        ptm_reference_params = deepcopy(self.param_handler(self.pretrained_model).get_ft_parameters())
        ftms_relevant_params = [ftm.get_ft_parameters() for ftm in self.ftms_params]
        ftms_task_dirs = self.get_task_directions(ptm_reference_params, ftms_relevant_params)

        ftms_task_mats = self.directions_to_matrices(ftms_task_dirs)
        ftms_others, ftms_mats = self.remove_others(ftms_task_mats)


        if self.merge_config['representation'] == 'cauchy_ebm':
            
            import time
            start = time.time()
            if self.merge_config["type"] == "fft":
                merged_task_vectors = self.get_merged_task_vectors_cauchy_merge_FFT(ftms_mats)
            elif self.merge_config["type"] == "lora":
                merged_task_vectors = self.get_merged_task_vectors_cauchy_merge_mode_LoRA(ftms_mats, 
                                                            tol=self.merge_config["tol"], 
                                                            jitter=self.merge_config["jitter"])
            end = time.time()
            print("Elapsed time:", end - start)
                
            
            
            # Name for load/save/plots
            if self.merge_config['merge_type'] == 'mode':
                name = f"{self.merge_config['type'].upper()}_CauchyEBM_MODE_jitter{self.merge_config['jitter']}_tol{self.merge_config['tol']}"
            else:
                name = f"{self.merge_config['type'].upper()}_CauchyEBM_LANG_steps{self.merge_config['steps']}_" + \
                    f"lr{self.merge_config['lr']}_temp{self.merge_config['temperature']}" + \
                    f"jitter{self.merge_config['jitter']}_tol{self.merge_config['tol']}_"+ \
                    f"noiseScale{self.merge_config['noise_scale']}_chains{self.merge_config['num_chains']}"
                


        # Save merged vector
        # torch.save(merged_task_vectors, f"logs/{self.merge_config['arch']}/7_tasks/" + name + ".pt")

        # Compute norm
        compute_norm(merged_task_vectors)
        self.ingredients = {
            'merged_task_vectors': merged_task_vectors,
        }

    def merge(self, merge_config={'merge_method': 'tv'}):

        merged_base = deepcopy(self.pretrained_model)

        merged_sd = self.ingredients["merged_task_vectors"]

        if self.merge_config['type'] == 'fft':
            merged_model = self.add_task_parameters_fft(merged_base, merged_sd, scaling_coeffs=self.merge_config['scaling_coeffs'])
        else:
            merged_model = self.add_task_parameters(merged_base, merged_sd, scaling_coeffs=self.merge_config['scaling_coeffs'])

        return merged_model

def get_merge_handler(rep_type):
    if rep_type == 'vector':
        return VectorMerger
    elif rep_type == 'cauchy_ebm':
        return CauchyEBMerger