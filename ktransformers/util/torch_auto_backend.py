import sys
import torch
from torch.utils.cpp_extension import CUDA_HOME
try:
    from torch_musa.utils.musa_extension import MUSA_HOME
except ImportError:
    MUSA_HOME=None

if CUDA_HOME is not None:
    CUDA = "cuda"
elif MUSA_HOME is not None:
    CUDA = "musa"

    torch.cuda = torch.musa
    torch.cuda.CUDAGraph = torch.musa.MUSAGraph

    # torch.cuda.nvtx.range_push = _pass_pvtx
    # torch.cuda.nvtx.range_pop = _pass_pvtx
    # torch.cuda.nvtx.range = _pass_pvtx
    # torch.cuda.is_available = torch.musa.is_available
    # torch.cuda.current_device = lambda : f'musa:{torch.musa.current_device()}'
    torch.cuda.device_count = torch.musa.device_count
    torch.cuda.set_device = torch.musa.set_device
    torch.cuda.DoubleTensor = torch.musa.DoubleTensor
    torch.cuda.FloatTensor = torch.musa.FloatTensor
    torch.cuda.LongTensor = torch.musa.LongTensor
    torch.cuda.HalfTensor = torch.musa.HalfTensor
    torch.cuda.BFloat16Tensor = torch.musa.BFloat16Tensor
    torch.cuda.IntTensor = torch.musa.IntTensor
    torch.cuda.synchronize = torch.musa.synchronize
    torch.cuda.get_rng_state = torch.musa.get_rng_state
    torch.cuda.set_rng_state = torch.musa.set_rng_state
    torch.cuda.synchronize = torch.musa.synchronize
    torch.cuda.empty_cache = torch.musa.empty_cache
    torch.Tensor.cuda = torch.Tensor.musa
    torch.cuda.manual_seed = torch.musa.manual_seed
    torch.cuda.Event = torch.musa.Event
    torch.cuda.Stream = torch.musa.Stream
    torch.cuda.get_device_properties = torch.musa.get_device_properties

     # Memory
    torch.cuda.memory_allocated = torch.musa.memory_allocated
    torch.cuda.max_memory_allocated = torch.musa.memory_allocated
    torch.cuda.memory_reserved = torch.musa.memory_reserved
    torch.cuda.max_memory_reserved = torch.musa.max_memory_reserved

    original_empty = torch.empty
    def patched_empty(*args, **kwargs):
        if 'device' in kwargs and kwargs['device'] == 'cuda':
            kwargs['device'] = 'musa'
        result = original_empty(*args, **kwargs)
        return result
    torch.empty = patched_empty

    torch.Tensor.double = torch.Tensor.float

    # **Monkey Patch `torch.Tensor.cuda()`**
    def tensor_cuda(self, device=None, non_blocking=False, memory_format=None):
        if device is None:
            device = CUDA
        elif isinstance(device, int):
            device = f"{CUDA}:{device}"
        return self.to(device, non_blocking=non_blocking, memory_format=memory_format)

    torch.Tensor.cuda = tensor_cuda

    # **Monkey Patch `torch.cuda.current_stream`**
    original_musa_current_stream = torch.musa.current_stream

    def patch_stream_object(stream):
        if not hasattr(stream, "cuda_stream"):
            stream.cuda_stream = stream.musa_stream
        return stream

    def patched_current_stream(device=None):
        return patch_stream_object(original_musa_current_stream(device))

    torch.cuda.current_stream = patched_current_stream

else:
    raise ValueError("Unsupported platform: {}".format(sys.platform))

CUDA0 = f"{CUDA}:0"
CUDA1 = f"{CUDA}:1"
CUDA2 = f"{CUDA}:2"

print(f"Torch backend loaded: CUDA={CUDA}, CUDA0={CUDA0}")