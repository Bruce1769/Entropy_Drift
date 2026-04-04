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

def extract_topk_logits(
    logits: torch.Tensor, topk: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Extract top-k logits and indices from a logits tensor.

    Args:
        logits: Unnormalized logits of shape [vocab_size] or [batch_size, vocab_size]
        topk: Number of top logits to keep

    Returns:
        Tuple of (topk_logits, topk_indices)
    """
    if topk <= 0:
        raise ValueError(f"topk must be positive, got {topk}")

    is_single_input = logits.dim() == 1
    if is_single_input:
        logits = logits.unsqueeze(0)

    k = min(int(topk), logits.shape[-1])
    topk_logits, topk_indices = torch.topk(logits, k=k, dim=-1)

    if is_single_input:
        return topk_logits.squeeze(0), topk_indices.squeeze(0)
    return topk_logits, topk_indices

def compute_sparse_topk_js_divergence(
    logits_p: torch.Tensor,
    indices_p: torch.Tensor,
    logits_q: torch.Tensor,
    indices_q: torch.Tensor,
) -> Union[float, torch.Tensor]:
    """
    Calculate sparse top-k JS divergence using the union of two top-k supports.

    Each side is treated as a truncated distribution over its own top-k support and
    is re-normalized on the union support before computing JS divergence.

    Args:
        logits_p: Top-k logits for distribution P, shape [k] or [batch_size, k]
        indices_p: Top-k indices for distribution P, same leading shape as logits_p
        logits_q: Top-k logits for distribution Q, shape [k] or [batch_size, k]
        indices_q: Top-k indices for distribution Q, same leading shape as logits_q

    Returns:
        JS divergence as a scalar (if single input) or tensor of shape [batch_size]
    """
    if logits_p.shape != indices_p.shape or logits_q.shape != indices_q.shape:
        raise ValueError(
            "Sparse logits and indices must have matching shapes on each side: "
            f"got {logits_p.shape}/{indices_p.shape} and {logits_q.shape}/{indices_q.shape}"
        )

    is_single_input = logits_p.dim() == 1
    if is_single_input:
        logits_p = logits_p.unsqueeze(0)
        indices_p = indices_p.unsqueeze(0)
        logits_q = logits_q.unsqueeze(0)
        indices_q = indices_q.unsqueeze(0)

    batch_size = logits_p.shape[0]
    js_values = []

    for i in range(batch_size):
        p_logits = logits_p[i].to(dtype=torch.float32)
        p_indices = indices_p[i].to(dtype=torch.long)
        q_logits = logits_q[i].to(dtype=torch.float32)
        q_indices = indices_q[i].to(dtype=torch.long)

        combined_indices = torch.cat([p_indices, q_indices], dim=0)
        union_indices, inverse = torch.unique(combined_indices, sorted=True, return_inverse=True)

        p_positions = inverse[: p_indices.shape[0]]
        q_positions = inverse[p_indices.shape[0] :]

        p_union_logits = torch.full(
            (union_indices.shape[0],),
            float("-inf"),
            dtype=torch.float32,
            device=p_logits.device,
        )
        q_union_logits = torch.full(
            (union_indices.shape[0],),
            float("-inf"),
            dtype=torch.float32,
            device=q_logits.device,
        )

        p_union_logits[p_positions] = p_logits
        q_union_logits[q_positions] = q_logits

        probs_p = torch.softmax(p_union_logits, dim=-1)
        probs_q = torch.softmax(q_union_logits, dim=-1)
        mean_probs = 0.5 * (probs_p + probs_q)
        log_mean_probs = torch.log(mean_probs.clamp_min(1e-12))

        log_probs_p = torch.log(probs_p.clamp_min(1e-12))
        log_probs_q = torch.log(probs_q.clamp_min(1e-12))
        kl_pm = torch.sum(probs_p * (log_probs_p - log_mean_probs))
        kl_qm = torch.sum(probs_q * (log_probs_q - log_mean_probs))
        js_values.append(0.5 * (kl_pm + kl_qm))

    js_divergence = torch.stack(js_values)
    return js_divergence[0].item() if is_single_input else js_divergence

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
