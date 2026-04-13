from typing import Dict, List

import torch
from torch import Tensor, nn
from tqdm import tqdm


def state_dicts_check_keys(state_dicts: List[Dict[str, Tensor]]):
    """
    Checks that the state dictionaries have the same keys.

    Args:
        state_dicts (List[Dict[str, Tensor]]): A list of dictionaries containing the state of PyTorch models.

    Raises:
        ValueError: If the state dictionaries have different keys.
    """
    # Get the keys of the first state dictionary in the list
    keys = set(state_dicts[0].keys())
    # Check that all the state dictionaries have the same keys
    for state_dict in state_dicts:
        assert keys == set(state_dict.keys()), "keys of state_dicts are not equal"


def num_params_of_state_dict(state_dict: Dict[str, Tensor]):
    """
    Returns the number of parameters in a state dict.

    Args:
        state_dict (Dict[str, Tensor]): The state dict to count the number of parameters in.

    Returns:
        int: The number of parameters in the state dict.
    """
    return sum([state_dict[key].numel() for key in state_dict])


def state_dict_flatten(state_dict: Dict[str, Tensor]):
    """
    Flattens a state dict.

    Args:
        state_dict (Dict[str, Tensor]): The state dict to be flattened.

    Returns:
        Tensor: The flattened state dict.
    """
    flattened_state_dict = []
    for key in state_dict:
        flattened_state_dict.append(state_dict[key].flatten())
    return torch.cat(flattened_state_dict)


def state_dict_avg(state_dicts: List[Dict[str, Tensor]]):
    """
    Returns the average of a list of state dicts.

    Args:
        state_dicts (List[Dict[str, Tensor]]): The list of state dicts to average.

    Returns:
        Dict: The average of the state dicts.
    """
    assert len(state_dicts) > 0, "The number of state_dicts must be greater than 0"
    assert all([len(state_dicts[0]) == len(state_dict) for state_dict in state_dicts]), "All state_dicts must have the same number of keys"

    avg_state_dict = {}
    for key in state_dicts[0]:
        avg_state_dict[key] = torch.zeros_like(state_dicts[0][key])
        for state_dict in state_dicts:
            avg_state_dict[key] += state_dict[key]
        avg_state_dict[key] /= len(state_dicts)
        # avg_state_dict[key] *= 1/8
    return avg_state_dict


def state_dict_sub(a: Dict, b: Dict, strict: bool = True):
    """
    Returns the difference between two state dicts.

    Args:
        a (Dict): The first state dict.
        b (Dict): The second state dict.
        strict (bool): Whether to check if the keys of the two state dicts are the same.

    Returns:
        Dict: The difference between the two state dicts.
    """
    if strict:
        assert set(a.keys()) == set(b.keys())
    diff = {}
    for k in a:
        if k in b:
            diff[k] = a[k] - b[k]
    return diff


def state_dict_add(a: Dict, b: Dict, strict: bool = True):
    """
    Returns the sum of two state dicts.

    Args:
        a (Dict): The first state dict.
        b (Dict): The second state dict.
        strict (bool): Whether to check if the keys of the two state dicts are the same.

    Returns:
        Dict: The sum of the two state dicts.
    """
    if strict:
        assert set(a.keys()) == set(b.keys())

    diff = {}
    for k in a:
        if k in b:
            diff[k] = a[k] + b[k]
    return diff


def state_dict_mul(state_dict: Dict, scalar: float):
    """
    Returns the product of a state dict and a scalar.

    Args:
        state_dict (Dict): The state dict to be multiplied.
        scalar (float): The scalar to multiply the state dict with.

    Returns:
        Dict: The product of the state dict and the scalar.
    """
    diff = {}
    for k in state_dict:
        diff[k] = scalar * state_dict[k]
    return diff


def state_dict_power(state_dict: Dict[str, Tensor], p: float):
    """
    Returns the power of a state dict.

    Args:
        state_dict (Dict[str, Tensor]): The state dict to be powered.
        p (float): The power to raise the state dict to.

    Returns:
        Dict[str, Tensor]: The powered state dict.
    """
    powered_state_dict = {}
    for key in state_dict:
        powered_state_dict[key] = state_dict[key] ** p
    return powered_state_dict


def state_dict_interpolation(state_dicts: List[Dict[str, Tensor]], scalars: List[float]):
    """
    Interpolates between a list of state dicts using a list of scalars.

    Args:
        state_dicts (List[Dict[str, Tensor]]): The list of state dicts to interpolate between.
        scalars (List[float]): The list of scalars to use for interpolation.

    Returns:
        Dict: The interpolated state dict.
    """
    assert len(state_dicts) == len(scalars), "The number of state_dicts and scalars must be the same"
    assert len(state_dicts) > 0, "The number of state_dicts must be greater than 0"
    assert all([len(state_dicts[0]) == len(state_dict) for state_dict in state_dicts]), "All state_dicts must have the same number of keys"

    interpolated_state_dict = {}
    for key in state_dicts[0]:
        interpolated_state_dict[key] = torch.zeros_like(state_dicts[0][key])
        for state_dict, scalar in zip(state_dicts, scalars):
            interpolated_state_dict[key] += scalar * state_dict[key]
    return interpolated_state_dict


def state_dict_sum(state_dicts: List[Dict[str, Tensor]]):
    """
    Returns the sum of a list of state dicts.

    Args:
        state_dicts (List[Dict[str, Tensor]]): The list of state dicts to sum.

    Returns:
        Dict: The sum of the state dicts.
    """
    assert len(state_dicts) > 0, "The number of state_dicts must be greater than 0"
    assert all([len(state_dicts[0]) == len(state_dict) for state_dict in state_dicts]), "All state_dicts must have the same number of keys"

    sum_state_dict = {}
    for key in state_dicts[0]:
        sum_state_dict[key] = torch.zeros_like(state_dicts[0][key])
        for state_dict in state_dicts:
            sum_state_dict[key] += state_dict[key]
    return sum_state_dict


def state_dict_weighted_sum(state_dicts: List[Dict[str, Tensor]], weights: List[float]):
    """
    Returns the weighted sum of a list of state dicts.

    Args:
        state_dicts (List[Dict[str, Tensor]]): The list of state dicts to interpolate between.
        weights (List[float]): The list of weights to use for the weighted sum.

    Returns:
        Dict: The weighted sum of the state dicts.
    """
    assert len(state_dicts) == len(weights), "The number of state_dicts and weights must be the same"
    assert len(state_dicts) > 0, "The number of state_dicts must be greater than 0"
    assert all([len(state_dicts[0]) == len(state_dict) for state_dict in state_dicts]), "All state_dicts must have the same number of keys"

    weighted_sum_state_dict = {}
    for key in state_dicts[0]:
        weighted_sum_state_dict[key] = torch.zeros_like(state_dicts[0][key])
        for state_dict, weight in zip(state_dicts, weights):
            weighted_sum_state_dict[key] += weight * state_dict[key]
    return weighted_sum_state_dict



def state_dict_cauchy(state_dicts, max_iters=50, tol=1e-4, jitter = 1e-3, eps=0.3, max_tries = 5, common_space=False, scale_radius=1000, init_zero=False):
        layers = state_dicts[0].keys()
        merged_weights = {}
        num_tasks = len(state_dicts)
        for layer_key in tqdm(list(layers), desc="Iterative closed-form merging"):
            # If not matrix then just average
            if len(state_dicts[0][layer_key].shape) != 2:
                merged_weights[layer_key] = torch.stack([state_dicts[i][layer_key] for i in range(num_tasks)], dim=0).mean(dim=0)
                print(f"Averaging for layer {layer_key} with shape {state_dicts[0][layer_key].shape}")
                continue

            # Stack all task matrices: (N, dim, dim)
            A_stack = torch.stack([state_dicts[i][layer_key] for i in range(num_tasks)], dim=0)
            # A_stack = A_stack.to(self.device)
            N, dim, _ = A_stack.shape
            
            if init_zero:
                A_T = A_stack.transpose(1, 2)
                ATA_stack = torch.bmm(A_stack, A_T)                  # (N, d, d)
                ATATA_stack = torch.bmm(ATA_stack, A_stack)          # (N, d, d)
                merged = torch.zeros_like(state_dicts[0][layer_key])
            else:
                # Initial guess: uniform merge using original method
                A_T = A_stack.transpose(1, 2)
                ATA_stack = torch.bmm(A_stack, A_T)                  # (N, d, d)
                H_init = ATA_stack.sum(dim=0) + 1e-3 * torch.eye(dim).to(ATA_stack.device) # (d, d)
                ATATA_stack = torch.bmm(ATA_stack, A_stack)          # (N, d, d)
                b_init = ATATA_stack.sum(dim=0)                     # (d, d)

                L_init = torch.linalg.cholesky(H_init)
                merged = torch.cholesky_solve(b_init, L_init)

            # Compute max task norm for ball radius
            task_norms = torch.norm(A_stack, p='fro', dim=(1,2))  # (N,)
            max_task_norm = task_norms.max().item()
            ball_radius = scale_radius * max_task_norm 

            for iter_idx in range(max_iters):
                # Compute diff = merged - A_i
                diff = merged - A_stack  # broadcasting

                # Compute A_i^T @ diff
                A_T_diff = torch.bmm(A_stack.transpose(1, 2), diff)
                # ||A_i^T @ (merged - A_i)||_F^2
                quad_term = (A_T_diff ** 2).sum(dim=(1, 2))  # (N,)
                # ||A_i||_F^2
                task_norm_sq = (A_stack ** 2).sum(dim=(1, 2))  # (N,)
                # denom = ((θ_i − θ_m)^T θ_i)^2 + eps
                # denom = quad_term + eps
                denom = quad_term + eps * task_norm_sq

                # alpha_i = 1 / denom
                alpha = 1.0 / denom  # (N,)

                # Compute weighted H = sum alpha_i * (A_i @ A_i^T) → (d, d)
                weighted_ATA = ATA_stack * alpha.view(-1, 1, 1)      # (N, d, d) * (N,1,1)
                H = weighted_ATA.sum(dim=0)

                # Compute weighted b = sum alpha_i * (A_i @ A_i^T @ A_i) → (d, d)
                weighted_ATATA = ATATA_stack * alpha.view(-1, 1, 1)
                b = weighted_ATATA.sum(dim=0)                        # (d, d)

                # Use Cholesky
                eye = torch.eye(H.shape[-1], device=H.device, dtype=H.dtype)
                jitter=1e-3
                H = H + jitter * eye
                for k in range(max_tries):
                    try:
                        L = torch.linalg.cholesky(H)
                        break
                    except RuntimeError:
                        jitter *= 10
                        # print("new jitter", jitter)
                        H = H + jitter * eye
                else:
                    raise RuntimeError(f"Cholesky failed after {max_tries} jitter retries")
                
                new_merged = torch.cholesky_solve(b, L)

                
                # Convergence
                change = torch.norm(new_merged - merged, p='fro').item()

                merged = new_merged

                if change < tol:
                    print(f"Layer {layer_key}: converged at iter {iter_idx+1} (change={change:.6f})")
                    break
            else:
                print(f"Layer {layer_key}: max iters reached (final change={change:.6f})")


            merged_weights[layer_key] = merged

        return merged_weights