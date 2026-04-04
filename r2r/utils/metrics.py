import torch
import torch.nn.functional as F
from typing import Tuple, Union

def compute_entropy(logits: torch.Tensor) -> Union[float, torch.Tensor]:
    """
    Calculate entropy of the prediction distribution.
    
    Args:
        logits: Unnormalized logits of shape [vocab_size] or [batch_size, vocab_size]
        
    Returns:
        Entropy values as a scalar (if single input) or tensor of shape [batch_size]
    """
    # Handle single dimension input
    is_single_input = logits.dim() == 1
    if is_single_input:
        logits = logits.unsqueeze(0)
    
    probs = F.softmax(logits, dim=-1)
    log_probs = F.log_softmax(logits, dim=-1)
    entropy = -torch.sum(probs * log_probs, dim=-1)  # [batch_size]
    
    return entropy.item() if is_single_input else entropy

def compute_logu(logits: torch.Tensor, topk: int = 10) -> Tuple[Union[float, torch.Tensor], Union[float, torch.Tensor]]:
    """
    Calculate log-u score of the prediction distribution.
    
    Args:
        logits: Unnormalized logits of shape [vocab_size] or [batch_size, vocab_size]
        topk: Number of top logits to consider
        
    Returns:
        Tuple of (aleatoric_uncertainty, epistemic_uncertainty)
        Each is a scalar (if single input) or tensor of shape [batch_size]
    """
    # Handle single dimension input
    is_single_input = logits.dim() == 1
    if is_single_input:
        logits = logits.unsqueeze(0)
    
    # Get top-k logits and their indices
    topk_logits, topk_indices = torch.topk(logits, topk, dim=-1)  # [batch_size, topk]
    
    # Calculate sum of logits (S)
    alpha = torch.sum(topk_logits, dim=-1, keepdim=True)  # [batch_size, 1]
    
    # Calculate normalized probabilities (p_i = x_i/S)
    probs = topk_logits / alpha  # [batch_size, topk]
    
    # Calculate digamma terms
    digamma_xi = torch.digamma(topk_logits + 1)  # ψ(x_i + 1)
    digamma_sum = torch.digamma(alpha + 1)  # ψ(S + 1)
    
    # Calculate aleatoric uncertainty efficiently
    # AU = -∑(p_i * (ψ(x_i + 1) - ψ(S + 1)))
    aleatoric_uncertainty = -torch.sum(probs * (digamma_xi - digamma_sum), dim=-1)  # [batch_size]
    
    # Calculate epistemic uncertainty
    # EU = K / (S + K)
    epistemic_uncertainty = topk / (alpha.squeeze(-1) + topk)  # [batch_size]
    
    if is_single_input:
        return aleatoric_uncertainty.item(), epistemic_uncertainty.item()
    else:
        return aleatoric_uncertainty, epistemic_uncertainty

def compute_js_divergence(
    logits_p: torch.Tensor, logits_q: torch.Tensor
) -> Union[float, torch.Tensor]:
    """
    Calculate the Jensen-Shannon divergence between two logit tensors.

    Args:
        logits_p: Unnormalized logits of shape [vocab_size] or [batch_size, vocab_size]
        logits_q: Unnormalized logits with the same shape as logits_p

    Returns:
        JS divergence as a scalar (if single input) or tensor of shape [batch_size]
    """
    if logits_p.shape != logits_q.shape:
        raise ValueError(
            f"logits_p and logits_q must have the same shape, got {logits_p.shape} and {logits_q.shape}"
        )

    is_single_input = logits_p.dim() == 1
    if is_single_input:
        logits_p = logits_p.unsqueeze(0)
        logits_q = logits_q.unsqueeze(0)

    logits_p = logits_p.to(dtype=torch.float32)
    logits_q = logits_q.to(dtype=torch.float32)

    log_probs_p = F.log_softmax(logits_p, dim=-1)
    log_probs_q = F.log_softmax(logits_q, dim=-1)
    probs_p = log_probs_p.exp()
    probs_q = log_probs_q.exp()

    mean_probs = 0.5 * (probs_p + probs_q)
    log_mean_probs = torch.log(mean_probs.clamp_min(1e-12))

    kl_pm = torch.sum(probs_p * (log_probs_p - log_mean_probs), dim=-1)
    kl_qm = torch.sum(probs_q * (log_probs_q - log_mean_probs), dim=-1)
    js_divergence = 0.5 * (kl_pm + kl_qm)

    return js_divergence.item() if is_single_input else js_divergence

def compute_reliability(logits: torch.Tensor, topk: int = 10) -> Union[float, torch.Tensor]:
    """
    Calculate reliability of the prediction distribution.
    
    Args:
        logits: Unnormalized logits of shape [vocab_size] or [batch_size, vocab_size]
        topk: Number of top logits to consider
        
    Returns:
        Reliability values as a scalar (if single input) or tensor of shape [batch_size]
    """
    aleatoric_uncertainty, epistemic_uncertainty = compute_logu(logits, topk)
    
    # Handle both scalar and tensor inputs
    if isinstance(aleatoric_uncertainty, float):
        return 1 / (aleatoric_uncertainty * epistemic_uncertainty)
    else:
        return 1 / (aleatoric_uncertainty * epistemic_uncertainty)
