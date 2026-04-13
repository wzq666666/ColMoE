import torch
import gc
import psutil
import os
import time
import threading
from multiprocessing import Process

def clean_mem():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif hasattr(torch_npu, 'npu'):
        torch_npu.npu.empty_cache() 

def print_mem_usage(tag=''):
    proc = psutil.Process(os.getpid())
    mem = proc.memory_info().rss / 1024 / 1024
    print(f"[{tag}] 当前内存使用: {mem:.2f} MB")

def get_mem_free():
    mem = psutil.virtual_memory()
    mem_free = mem.available / 1024 / 1024  # 转换为 MB
    return mem_free

def memory_and_swap_guard(mem_thresh=300, swap_thresh=0, always_print=False):  # 单位 MB
    while True:
        mem = psutil.virtual_memory()
        swap = psutil.swap_memory()
        mem_free = mem.available / 1024 / 1024
        swap_free = swap.free / 1024 / 1024
        if always_print:
            print(f"[MemoryGuard] MemFree={mem_free:.1f}MB, SwapFree={swap_free:.1f}MB")
        if mem_free < mem_thresh or swap_free < swap_thresh:
            print(f"⚠️ MemFree={mem_free:.1f}MB, SwapFree={swap_free:.1f}MB — 即将退出")
            os._exit(1)  # 强制退出推理
        time.sleep(0.1)

def launch_memory_guard():
    t = threading.Thread(target=memory_and_swap_guard, daemon=True)
    t.start()

