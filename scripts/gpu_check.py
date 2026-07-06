import torch
print("torch", torch.__version__, "| built-cuda", torch.version.cuda, flush=True)
print("AVAIL", torch.cuda.is_available(), "COUNT", torch.cuda.device_count(), flush=True)
try:
    torch.cuda.init()
    print("init OK", flush=True)
except Exception as e:
    print("init FAIL:", repr(e)[:200], flush=True)
