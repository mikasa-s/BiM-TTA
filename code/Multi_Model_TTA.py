import time
from copy import deepcopy
import torch
import torch.nn as nn
import torch.jit
from torch.cuda.amp import autocast, GradScaler
import math
import torch.nn.functional as F
import numpy as np


class Multi_Modal_TTA(nn.Module):
    def __init__(self, model, optimizer, device, args, steps=1, episodic=False):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.steps = steps
        assert steps > 0, "TTA requires >=1 steps for forward and update"
        self.episodic = episodic
        self.args = args
        self.device = device

    def forward(self, x, adapt_flag):
        if adapt_flag:
            outputs, _, _, _, loss = forward_and_adapt(
                x, self.model, self.optimizer, self.args)
        else:
            outputs, _, _, _ = self.model.forward(x)
            loss = (0, 0)
            outputs = (outputs, outputs)

        return outputs, loss


@torch.enable_grad()
def forward_and_adapt(x, model, optimizer, args):
    """
    Forward pass and adapt model on batch of data.
    Computes mutual information sharing using filtered modality data while returning final filtered indices.
    """
    if torch.cuda.is_available():
        x = x.cuda()

    joint_probs_o, modal_1_probs_o, modal_2_probs_o, modal_3_probs_o, Eembedding_1, Eembedding_2, Eembedding_3 = model.forward_eval(x)

    total_iterations = 7
    beta = 0.2

    participated_indices = torch.zeros(x.shape[0], dtype=torch.bool, device=x.device)

    for current_iteration in range(1, total_iterations + 1):
        print(f'current_iteration: {current_iteration}')

        filtered_list = ~participated_indices

        if not filtered_list.any():
            print("All samples have already participated, terminating early.")
            break

        filtered_x = x[filtered_list]
        joint_probs, modal_1_probs, modal_2_probs, modal_3_probs, _, _, _ = model.forward_eval(filtered_x)

        mu_1 = 0.3
        mu_2 = 0.2
        var_coefficient = 2.0
        beta_t = beta + (1 - beta) * current_iteration / total_iterations

        joint_entropy = entropy(joint_probs)

        modal_entropies = [entropy(modality_probs) for modality_probs in [modal_1_probs, modal_2_probs, modal_3_probs]]
        weighted_modal_entropy = modal_entropies[0] + mu_1 * modal_entropies[1] + mu_2 * modal_entropies[2]

        joint_entropy_mean = joint_entropy.mean().item()
        joint_entropy_var = joint_entropy.var().item()

        modal_entropy_mean = [modal_entropy.mean().item() for modal_entropy in modal_entropies]
        modal_entropy_var = [modal_entropy.var().item() for modal_entropy in modal_entropies]

        bound_m = joint_entropy_mean + var_coefficient * beta_t * joint_entropy_var
        bound_u = (
            (modal_entropy_mean[0] + mu_1 * modal_entropy_mean[1] + mu_2 * modal_entropy_mean[2])
            - var_coefficient * beta_t * (modal_entropy_var[0] + mu_1 * modal_entropy_var[1] + mu_2 * modal_entropy_var[2])
        )

        entropy_indices = (joint_entropy <= bound_m) & (weighted_modal_entropy >= bound_u)
        filtered_list[filtered_list.clone()] = entropy_indices

        if not filtered_list.any():
            print("No samples remain after multimodal/unimodal entropy filtering, skipping to next iteration.")
            continue
        print(f"Remaining samples after filtering: {filtered_list.sum().item()}")

        participated_indices[filtered_list] = True

        filtered_x = x[filtered_list]
        joint_probs, modal_1_probs, modal_2_probs, modal_3_probs, _, _, _ = model.forward_eval(filtered_x)

        num_modalities = 3
        modal_logits = [modal_1_probs, modal_2_probs, modal_3_probs]
        complementary_probs = []
        for i in range(num_modalities):
            other_probs = torch.stack([modal_logits[j] for j in range(num_modalities) if j != i], dim=0)
            complementary_prob = other_probs.mean(dim=0)
            complementary_probs.append(complementary_prob)

        mutual_info_loss = 0
        for i in range(num_modalities):
            p_u = modal_logits[i]
            p_u_prime = complementary_probs[i]
            avg_prob = (p_u_prime + joint_probs) / 2
            loss_mis = kl_divergence(p_u, avg_prob).mean()
            mutual_info_loss += loss_mis

        Ent_theta = entropy(joint_probs)
        Ent_theta_0 = 0.4 * torch.log(torch.tensor(2, dtype=torch.float32))
        alpha = 1 / torch.exp(Ent_theta - Ent_theta_0)

        lambda_tradeoff = 1
        t_bound = 1
        t0 = round(total_iterations * t_bound)
        if current_iteration <= t0:
            loss = (alpha * (Ent_theta + lambda_tradeoff * mutual_info_loss)).mean()
        else:
            loss = (alpha * Ent_theta).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if participated_indices.sum().item() == 0:
        loss = torch.tensor(0.0)
    print(f'total participated samples: {participated_indices.sum().item()}')

    with torch.no_grad():
        joint_probs_, modal_1_probs_, modal_2_probs_, modal_3_probs_, _, _, _ = model.forward_eval(x)

    return (
        (joint_probs_o, joint_probs_),
        (modal_1_probs_o, modal_1_probs_),
        (modal_2_probs_o, modal_2_probs_),
        (modal_3_probs_o, modal_3_probs_),
        loss.item(),
    )


def kl_divergence(p, q):
    return torch.sum(p * torch.log((p + 1e-9) / (q + 1e-9)), dim=1)


def entropy(p):
    """
    Calculate the entropy for each sample.
    :param p: Tensor of shape [batch_size, num_features], representing sample features or distribution
    :return: Tensor of shape [batch_size], containing entropy values for each sample
    """
    return -torch.sum(p * torch.log(p + 1e-9), dim=-1)


def get_params():
    BN_modules = [
        'encoder1.module.branch1.1',
        'encoder1.module.branch1.4.fusion_layer.0',
        'encoder1.module.branch1.5',
        'encoder1.module.branch2.1',
        'encoder1.module.branch2.4.fusion_layer.0',
        'encoder1.module.branch2.5',
        'encoder1.module.branch3.1',
        'encoder1.module.branch3.4.fusion_layer.0',
        'encoder1.module.branch3.5',
        'encoder2.module.branch1.1',
        'encoder2.module.branch1.5',
        'encoder2.module.branch2.1',
        'encoder2.module.branch2.5',
        'encoder2.module.branch3.1',
        'encoder2.module.branch3.5',
        'encoder3.module.branch1.1',
        'encoder3.module.branch1.5',
        'encoder3.module.branch2.1',
        'encoder3.module.branch2.5',
        'encoder3.module.branch3.1',
        'encoder3.module.branch3.5',
    ]

    CONV_modules = [
        'encoder1.module.branch1.0',
        'encoder1.module.branch2.0',
        'encoder1.module.branch3.0',
        'encoder2.module.branch1.0',
        'encoder2.module.branch2.0',
        'encoder2.module.branch3.0',
        'encoder3.module.branch1.0',
        'encoder3.module.branch2.0',
        'encoder3.module.branch3.0',
    ]

    SF_modules = [
        'SF.in_proj'
    ]

    all_modules = BN_modules + CONV_modules + SF_modules

    return BN_modules, CONV_modules, SF_modules, all_modules


def collect_BN_params(model):
    """
    Iterate through model modules to collect BatchNorm parameters.
    Returns parameters and their names.
    Note: Other parameterization choices are possible!
    """
    params_BN = []
    names_BN = []

    BN_modules, _, _, _ = get_params()

    for nm, m in model.named_modules():
        if nm in BN_modules:
            for np, p in m.named_parameters():
                if np in ['weight', 'bias']:
                    params_BN.append(p)
                    names_BN.append(f"{nm}.{np}")

    return params_BN, names_BN


def collect_CONV_params(model):
    """
    Iterate through model modules to collect convolution parameters.
    Returns parameters and their names.
    Note: Other parameterization choices are possible!
    """
    params_CONV = []
    names_CONV = []

    _, conv_modules, _, _ = get_params()

    for nm, m in model.named_modules():
        if nm in conv_modules:
            for np, p in m.named_parameters():
                if np in ['weight', 'bias']:
                    params_CONV.append(p)
                    names_CONV.append(f"{nm}.{np}")

    return params_CONV, names_CONV


def collect_SF_params(model):
    """
    Iterate through model modules to collect fusion parameters.
    Returns parameters and their names.
    """
    params_fusion = []
    names_fusion = []

    _, _, sf_modules, _ = get_params()

    for nm, m in model.named_modules():
        if nm in sf_modules:
            for np, p in m.named_parameters():
                if np in ['weight', 'bias']:
                    params_fusion.append(p)
                    names_fusion.append(f"{nm}.{np}")

    return params_fusion, names_fusion


def copy_model_and_optimizer(model, optimizer):
    """Copy model and optimizer states for reset after adaptation."""
    model_state = deepcopy(model.state_dict())
    optimizer_state = deepcopy(optimizer.state_dict())
    return model_state, optimizer_state


def load_model_and_optimizer(model, optimizer, model_state, optimizer_state):
    """Restore model and optimizer states from backup."""
    model.load_state_dict(model_state, strict=True)
    optimizer.load_state_dict(optimizer_state)


def configure_model(model):
    """Configure model for Renata adaptation."""
    model.train()
    model.requires_grad_(False)

    _, _, _, all_modules = get_params()

    for nm, m in model.named_modules():
        if nm in all_modules:
            m.requires_grad_(True)

    return model
