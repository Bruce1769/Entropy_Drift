import os

import torch
from sglang.srt.layers.activation import GeluAndMul, SiluAndMul
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
from sglang.srt.layers.attention.flashinfer_backend import FlashInferAttnBackend
from sglang.srt.layers.layernorm import GemmaRMSNorm, RMSNorm
from sglang.srt.layers.rotary_embedding import RotaryEmbedding

from r2r.models.sglang_patch.scheduler import __init__ as _init_scheduler, get_next_batch_to_run, get_new_batch_prefill, check_batch_status
from r2r.models.sglang_patch.schedule_batch import __init__ as _init_req, init_next_round_input, reset_for_retract, prepare_for_extend, filter_batch
from r2r.models.sglang_patch.flashinfer_cuda_graph import __init__ as _init_flashinfer_cuda_graph, init_forward_metadata_capture_cuda_graph, init_forward_metadata_replay_cuda_graph, forward_extend

def _should_fallback_layernorm_native() -> bool:
    force = os.environ.get("R2R_FORCE_NATIVE_RMSNORM")
    if force is not None:
        return force.strip().lower() not in ("0", "false", "no")
    try:
        return torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 12
    except Exception:
        return False

if _should_fallback_layernorm_native():
    RMSNorm.forward_cuda = RMSNorm.forward_native
    GemmaRMSNorm.forward_cuda = GemmaRMSNorm.forward_native


def _should_fallback_elementwise_native() -> bool:
    force = os.environ.get("R2R_FORCE_NATIVE_SGL_KERNEL")
    if force is not None:
        return force.strip().lower() not in ("0", "false", "no")
    try:
        return torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 12
    except Exception:
        return False


if _should_fallback_elementwise_native():
    RotaryEmbedding.forward_cuda = RotaryEmbedding.forward_native
    SiluAndMul.forward_cuda = SiluAndMul.forward_native
    GeluAndMul.forward_cuda = GeluAndMul.forward_native

FlashInferAttnBackend.__init__ = _init_flashinfer_cuda_graph
FlashInferAttnBackend.init_forward_metadata_capture_cuda_graph = init_forward_metadata_capture_cuda_graph
FlashInferAttnBackend.init_forward_metadata_replay_cuda_graph = init_forward_metadata_replay_cuda_graph
FlashInferAttnBackend.forward_extend = forward_extend

Scheduler.__init__ = _init_scheduler
Scheduler.get_next_batch_to_run = get_next_batch_to_run
Scheduler.get_new_batch_prefill = get_new_batch_prefill
Scheduler.check_batch_status = check_batch_status

ScheduleBatch.prepare_for_extend = prepare_for_extend
ScheduleBatch.filter_batch = filter_batch

Req.__init__ = _init_req
Req.init_next_round_input = init_next_round_input
Req.reset_for_retract = reset_for_retract
