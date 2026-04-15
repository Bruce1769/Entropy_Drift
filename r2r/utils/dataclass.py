import torch
from dataclasses import dataclass
from typing import List, Optional, Any

@dataclass
class ModelOutputs:
    """
    Outputs from the model
    
    Args:
        logits: shape (batch_size, seq_len, vocab_size)
        hidden_states: shape (batch_size, seq_len, hidden_size), as a list of tensors, with the last item being the last layer
        token: shape (batch_size, seq_len), the token that was used to generate the output
        reference_logits: optional reference-model logits for the same token positions
        reference_topk_indices: optional reference-model top-k indices for sparse routing
        reference_topk_logits: optional reference-model top-k logits for sparse routing
        sequence_ids: optional stable request identifiers aligned with batch rows
        positions: optional generated token positions aligned with batch rows
    """
    logits: torch.Tensor
    hidden_states: List[torch.Tensor]
    token: torch.Tensor
    reference_logits: Optional[torch.Tensor] = None
    reference_topk_indices: Optional[torch.Tensor] = None
    reference_topk_logits: Optional[torch.Tensor] = None
    sequence_ids: Optional[List[Any]] = None
    positions: Optional[List[int]] = None
