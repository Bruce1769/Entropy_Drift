import torch
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class ModelOutputs:
    """
    Outputs from the model
    
    Args:
        logits: shape (batch_size, seq_len, vocab_size)
        hidden_states: shape (batch_size, seq_len, hidden_size), as a list of tensors, with the last item being the last layer
        token: shape (batch_size, seq_len), the token that was used to generate the output
        reference_logits: optional reference-model logits for the same token positions
    """
    logits: torch.Tensor
    hidden_states: List[torch.Tensor]
    token: torch.Tensor
    reference_logits: Optional[torch.Tensor] = None
