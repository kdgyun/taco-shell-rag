def require_cuda(torch):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Check the NVIDIA driver and container GPU access.")


def check_loaded_model(model, precision):
    if model.device.type != "cuda":
        raise RuntimeError("a model is not on CUDA; CPU fallback is disabled")
    if str(model.dtype).removeprefix("torch.") != precision:
        raise RuntimeError("the requested model precision was not applied")
