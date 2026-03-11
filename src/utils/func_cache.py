import functools
import threading
import torch


def cache_tensor(func):
    cache = {}
    lock = threading.Lock()
    sentinel = object()

    @functools.wraps(func)
    def wrapper(tensorlike_arg):
        # key 计算在锁外进行（可能涉及 GPU 操作，耗时较长）
        key = (*tensorlike_arg.shape, torch.sin(tensorlike_arg).sum().item(), torch.cosh(tensorlike_arg).sum().item())
        with lock:
            result = cache.get(key, sentinel)
            if result is sentinel:
                result = func(tensorlike_arg)
                cache[key] = result
        return result

    return wrapper
